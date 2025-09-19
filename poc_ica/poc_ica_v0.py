# -*- coding: utf-8 -*-
# @Time    : 2025/9/8
# @Author  : dizjia
# @File    : poc_ica_v0.py
# @Description: 孵化了idea雏形🐣
#               内部一致性分析最初的详细注释版本。
#               仅用于学习


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random
import copy

# --- 1. 环境设置与模型定义 ---

# 为保证实验可复现，设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 定义一个简单的CNN模型用于CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        features = self.conv_stack(x)
        penultimate_features = self.fc_stack(features)
        logits = self.classifier(penultimate_features)
        return logits

    # 我们需要一个方法来提取倒数第二层的特征
    def get_penultimate_features(self, x):
        features = self.conv_stack(x)
        penultimate_features = self.fc_stack(features)
        return penultimate_features

# --- 2. 联邦学习与数据处理 ---

def get_cifar10_data():
    """下载并准备CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return train_dataset

def create_iid_partitions(dataset, num_clients):
    """将数据平均分配给客户端"""
    num_items = int(len(dataset) / num_clients)
    """
    dict_users存储每个客户端（key）以及分配给该客户端的数据索引集合（value）
    all_idxs是一个列表，存储数据集中所有数据的索引
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        """
        np.random.choice(...): 从当前的索引池 all_idxs 中无放回地随机抽取 num_items 个索引
        set(...): 将抽到的索引（一个Numpy数组）转换成一个集合
        """
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    """Subset根据给定的索引 list(idxs) 从原始的 dataset 中创建一个新的、更小的数据集子集"""
    return [Subset(dataset, list(idxs)) for _, idxs in dict_users.items()]

# --- 3. 后门攻击 (BadNets) 实现 ---

# 定义触发器
def get_trigger(img_size=32, trigger_size=4, position='bottom-right'):
    """
    创建一个白色的方块触发器
    生成一个特定尺寸的图像，其中大部分区域是黑色的，只有一个小的白色方块（即“触发器”）位于指定的位置
    0黑色，1白色
    """
    trigger = torch.zeros((3, img_size, img_size))
    if position == 'bottom-right':
        trigger[:, -trigger_size:, -trigger_size:] = 1.0
    return trigger

# 继承Dataset类，用于注入后门
class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, poison_ratio=0.1, target_label=0):
        self.original_dataset = original_dataset
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger = get_trigger()

        # 选定一部分数据进行投毒
        self.poison_indices = np.random.choice(
            len(self.original_dataset),
            int(len(self.original_dataset) * self.poison_ratio),
            replace=False
        )

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        if idx in self.poison_indices:
            # 注入触发器并修改标签
            img = torch.clamp(img + self.trigger, 0, 1) # clamp确保像素值在有效范围内
            label = self.target_label
        return img, label


# --- 4. 客户端和服务器的定义 ---

class Client:
    def __init__(self, client_id, local_data, is_malicious=False):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        if is_malicious:
            # 恶意客户端使用带毒的数据集
            # 在这里修改投毒率poison_ratio
            self.local_data = PoisonedDataset(local_data, poison_ratio=0.3, target_label=0)
        else:
            self.local_data = local_data
        # 在这里修改batch_size
        self.dataloader = DataLoader(self.local_data, batch_size=32, shuffle=True)

    def train(self, global_model_state_dict, local_epochs=1):
        """在本地数据上训练模型，并返回模型更新"""
        # 在这里修改本地训练轮次
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model_state_dict)
        # 在这里修改优化器和学习率
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 计算模型更新 (Δw)
        update = {key: model.state_dict()[key] - global_model_state_dict[key] for key in global_model_state_dict}
        return update

# --- 5. 核心：内部一致性分析实现 ---

def internal_consistency_analysis(global_model_state_dict, client_update, device):
    """
    对给定的客户端更新进行内部一致性分析。

    Args:
        global_model_state_dict (dict): 上一轮的全局模型状态字典。
        client_update (dict): 某个客户端上传的模型更新 (Δw)。
        device (torch.device): 计算设备 (mps或cuda)。

    Returns:
        float: 一致性得分（特征空间的距离），分数越高越可疑。
    """
    # 1. 创建一个应用了更新的临时模型
    temp_model = SimpleCNN().to(device)
    temp_model_state_dict = {key: global_model_state_dict[key] + client_update[key] for key in global_model_state_dict}
    temp_model.load_state_dict(temp_model_state_dict)
    temp_model.eval()

    # 2. 生成探测器对 (X, X')
    # X 是一个随机噪声图像
    probe_X = torch.randn(1, 3, 32, 32).to(device)
    # X' 是 X 的一个微小扰动版本
    # 在此修改扰动强度
    perturbation = torch.randn(1, 3, 32, 32).to(device) * 0.01 # 扰动强度很小
    probe_X_prime = probe_X + perturbation

    # 3. 提取特征并计算距离
    with torch.no_grad():
        features_X = temp_model.get_penultimate_features(probe_X)
        features_X_prime = temp_model.get_penultimate_features(probe_X_prime)

    # 使用余弦距离来衡量差异
    # 余弦距离 = 1 - 余弦相似度
    cosine_similarity = nn.functional.cosine_similarity(features_X, features_X_prime)
    consistency_score = 1 - cosine_similarity.item()

    return consistency_score


# --- 6. 实验主流程 ---

if __name__ == "__main__":
    print("--- 概念验证实验开始 ---")

    # 参数设置
    NUM_CLIENTS = 10
    MALICIOUS_CLIENTS = 2
    FL_ROUNDS = 5 # 运行几轮让模型初步收敛，在此处修改联邦学习轮次
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"使用设备: {DEVICE}")

    # 1. 准备数据和客户端
    train_dataset = get_cifar10_data()
    client_datasets = create_iid_partitions(train_dataset, NUM_CLIENTS)

    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i < MALICIOUS_CLIENTS # 前几个客户端是恶意的
        client = Client(i, client_datasets[i], is_malicious)
        clients.append(client)
    print(f"创建了 {NUM_CLIENTS} 个客户端, 其中 {MALICIOUS_CLIENTS} 个是恶意的。")

    # 2. 初始化全局模型
    global_model = SimpleCNN().to(DEVICE)
    global_model_state_dict = global_model.state_dict()

    # 3. 运行几轮完全正常的联邦学习，不加入攻击者
    # 让全局模型度过最混沌的训练初期，达到一个相对稳定和有意义的状态
    print(f"\n--- 进行 {FL_ROUNDS} 轮联邦学习以获得一个基础模型 ---")
    for round_num in range(FL_ROUNDS):
        print(f"第 {round_num + 1} 轮...")
        all_updates = []
        for client in clients:
            update = client.train(copy.deepcopy(global_model_state_dict))
            all_updates.append(update)

        # 简单的FedAvg聚合
        avg_update = {key: torch.stack([upd[key] for upd in all_updates], dim=0).mean(dim=0) for key in all_updates[0]}
        global_model_state_dict = {key: global_model_state_dict[key] + avg_update[key] for key in global_model_state_dict}
        global_model.load_state_dict(global_model_state_dict)

    print("基础模型训练完成。")

    # 4. 关键验证步骤：获取并分析一个良性和恶意更新
    # 基于上一阶段结束时的全局模型，让一个良性和恶意客户端分别进行一次本地训练
    print("\n--- 关键验证：分析一个良性更新和一个恶意更新 ---")

    # a. 获取一个恶意客户端的更新
    malicious_client = clients[0] # 我们知道第一个是恶意的
    malicious_update = malicious_client.train(copy.deepcopy(global_model_state_dict))

    # b. 获取一个良性客户端的更新
    benign_client = clients[-1] # 我们知道最后一个是良性的
    benign_update = benign_client.train(copy.deepcopy(global_model_state_dict))

    # c. 对两个更新分别进行内部一致性分析
    malicious_score = internal_consistency_analysis(copy.deepcopy(global_model_state_dict), malicious_update, DEVICE)
    benign_score = internal_consistency_analysis(copy.deepcopy(global_model_state_dict), benign_update, DEVICE)

    # 5. 打印结果
    print("\n--- 实验结果 ---")
    print(f"良性客户端更新的一致性得分: {benign_score:.6f}")
    print(f"恶意客户端更新的一致性得分: {malicious_score:.6f}")

    if malicious_score > benign_score * 2: # 设置一个简单的判断逻辑
        print("\n[结论]: 实验成功！恶意更新的一致性得分显著高于良性更新。")
        print("这验证了我们的核心假设：恶意更新会破坏模型的内部稳定性。")
    else:
        print("\n[结论]: 实验结果不显著。可能需要调整扰动强度、模型或训练轮次。")
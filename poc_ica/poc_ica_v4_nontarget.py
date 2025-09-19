# -*- coding: utf-8 -*-
# @Time    : 2025/9/11
# @Author  : dizjia
# @File    : poc_ica_v4_nontarget.py
# @Description: “不稳定增量分析”概念验证实验第四版的非目标攻击版。
#               核心修改：将攻击从BadNets修改为Label Flipping
#               为了保持这个文件夹的一致，文件命名暂时保持为ica不变
# 😁😁😁😁😁SUCCESS!!!


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import os
import time
import pathlib

# 基于传入的真实图片生成探测对

# 创建一个集中的配置字典来管理所有超参数
config = {
    # 联邦学习与环境设置
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "mps"),
    "SEED": 42,
    "NUM_CLIENTS": 10,
    "FL_ROUNDS": 120,

    # 客户端本地训练参数
    "LOCAL_EPOCHS": 2,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.01,

    # 学习率调度器 (Scheduler) 设置
    "SCHEDULER_ENABLED": True,
    "SCHEDULER_PATIENCE": 5,   # 连续5轮 test_loss 不下降就调整
    "SCHEDULER_FACTOR": 0.5,   # 学习率衰减为原来的一半 (lr = lr * 0.5)

    # 后门攻击参数
    "MALICIOUS_CLIENTS": 2,
    "POISON_RATIO": 0.3,

    # 内部一致性分析参数，这是我们的探测力度，越大探测力度越强
    "PERTURBATION_STRENGTH": 0.5,
    # "DISTANCE_METRIC": "cosine" # 未来可以扩展，例如 "l2"

    # 是否更改随机种子（使用随机图片进行一致性分析）
    "CHANGE_SEED": True,  # True 表示更改随机种子，使用随机图片进行一致性分析

    "NUM_PROBES": 10, # 使用10张图片进行集成探测

    # 是否早停
    "EARLY_STOPPING_ENABLED": True,      # True启用早停
    "EARLY_STOPPING_PATIENCE": 10,       # 连续10轮性能不提升就停止
    "EARLY_STOPPING_METRIC": "loss",     # 监控指标: 'loss' 或 'accuracy'

    "LOAD_SAVED_MODEL": True,  # 设置为True则跳过训练，直接加载模型进行分析
    "MODEL_SAVE_PATH": "./saved_model_pi/best_model_15.pth",
}


# --- 1. 环境设置与模型定义 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(config["SEED"])  # 全局固定种子是为了保证训练过程的可复现性


class SimpleCNN(nn.Module):
    # (这部分代码与原来完全相同，保持不变)
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

    def get_penultimate_features(self, x):
        features = self.conv_stack(x)
        penultimate_features = self.fc_stack(features)
        return penultimate_features


# --- 2. 联邦学习与数据处理 ---
def get_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 构建一个健壮的、与执行位置无关的数据目录路径
    # 1. 获取当前脚本文件 (poc_ica.py) 的路径
    current_file_path = pathlib.Path(__file__)
    # 2. 获取脚本文件所在的目录 (scripts/)
    script_dir = current_file_path.parent
    # 3. 获取项目根目录 (my_project/), 也就是脚本目录的上一级
    project_root = script_dir.parent
    # 4. 构造出 data 目录的绝对路径
    data_dir = project_root / "data"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def create_iid_partitions(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return [Subset(dataset, list(idxs)) for _, idxs in dict_users.items()]


# 创建一个用于“标签翻转”非目标攻击的数据集类
class LabelFlippingDataset(Dataset):
    def __init__(self, original_dataset, poison_ratio, num_classes=10):
        """
        初始化标签翻转数据集。
        Args:
            original_dataset (Dataset): 原始的、干净的数据集。
            poison_ratio (float): 数据集中被投毒（标签被翻转）的比例。
            num_classes (int): 数据集中的总类别数，对于CIFAR-10是10。
        """
        self.original_dataset = original_dataset
        self.poison_ratio = poison_ratio
        self.num_classes = num_classes

        # 选定一部分数据的索引进行投毒
        self.poison_indices = np.random.choice(
            len(self.original_dataset),
            int(len(self.original_dataset) * self.poison_ratio),
            replace=False
        )

        # 为了方便快速查找，将索引列表转换为集合
        self.poison_indices_set = set(self.poison_indices)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始的图像和标签
        img, original_label = self.original_dataset[idx]

        # 检查当前索引是否需要被投毒
        if idx in self.poison_indices_set:
            # --- 标签翻转的核心逻辑 ---
            # 创建一个包含了所有可能的错误标签的列表
            possible_wrong_labels = list(range(self.num_classes))
            possible_wrong_labels.remove(original_label)

            # 从错误标签列表中随机选择一个作为新的标签
            flipped_label = random.choice(possible_wrong_labels)

            return img, flipped_label
        else:
            # 如果不投毒，则返回原始的图像和标签
            return img, original_label

# --- 4. 客户端和服务器的定义 ---
class Client:
    def __init__(self, client_id, local_data, is_malicious=False):
        self.client_id = client_id
        self.device = config["DEVICE"]  ### 修改 ###

        if is_malicious:
            ### 修改 ###: 让恶意客户端使用 LabelFlippingDataset
            print(f"    (客户端 {client_id} 是恶意的，使用标签翻转攻击)")
            self.local_data = LabelFlippingDataset(
                local_data,
                poison_ratio=config["POISON_RATIO"],
                num_classes=10  # CIFAR-10 有10个类别
            )
        else:
            self.local_data = local_data

        self.dataloader = DataLoader(self.local_data, batch_size=config["BATCH_SIZE"], shuffle=True)

    def train(self, global_model_state_dict,learning_rate):
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        local_losses = []
        for epoch in range(config["LOCAL_EPOCHS"]):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                local_losses.append(loss.item())

        update = {key: model.state_dict()[key] - global_model_state_dict[key] for key in global_model_state_dict}
        return update, np.mean(local_losses)


def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


# --- 5. 核心：内部一致性分析实现 ---
def internal_consistency_analysis(global_model_state_dict, client_update, device, probe_image):
    """
        对给定的客户端更新进行内部一致性分析。

        核心修改：使用一张真实的图片作为探测基础，而非随机噪声。

        Args:
            global_model_state_dict (dict): 上一轮的全局模型状态字典。
            client_update (dict): 某个客户端上传的模型更新 (Δw)。
            device (torch.device): 计算设备 (cpu或cuda)。
            probe_image (torch.Tensor): 从数据集中抽取的、用于探测的3x32x32图像张量。

        Returns:
            float: 一致性得分（特征空间的距离），分数越高越可疑。
            当模型最稳定时，余弦相似度为 +1，我们的得分是 1 - 1 = 0。
            当模型最不稳定时，两个特征向量方向完全相反，余弦相似度为 -1，我们的得分是 1 - (-1) = 2。
            当两个特征向量正交时，余弦相似度为0，我们的得分是1 - 0 = 1。
            所以，我们这个“不一致性得分”的含义是：
            得分 = 0: 完美稳定。
            得分 = 1: 完全不相关。
            得分 = 2: 方向完全相反，极度不稳定。
        """
    temp_model = SimpleCNN().to(device)
    temp_model_state_dict = {key: global_model_state_dict[key] + client_update[key] for key in global_model_state_dict}
    temp_model.load_state_dict(temp_model_state_dict)
    temp_model.eval()

    # 2. 生成探测器对 (X, X')
    # 基于传入的真实图片生成探测对
    probe_X = probe_image.unsqueeze(0).to(device)  # probe_image 是 3x32x32, 增加一个批量维度

    # 扰动现在是加在真实图片上
    perturbation = torch.randn_like(probe_X) * config["PERTURBATION_STRENGTH"]
    probe_X_prime = probe_X + perturbation

    # 3. 提取特征并计算距离 (后续不变)
    with torch.no_grad():
        features_X = temp_model.get_penultimate_features(probe_X)
        features_X_prime = temp_model.get_penultimate_features(probe_X_prime)

    cosine_similarity = nn.functional.cosine_similarity(features_X, features_X_prime)
    consistency_score = 1 - cosine_similarity.item()

    return consistency_score


# --- 6. 实验主流程 ---
if __name__ == "__main__":
    print("--- 概念验证实验开始 ---")

    # 参数设置部分已被移至顶部的 config 字典
    DEVICE = config["DEVICE"]  ### 修改 ###
    print(f"使用设备: {DEVICE}")

    # 准备数据和客户端
    train_dataset, test_dataset = get_cifar10_data()
    client_datasets = create_iid_partitions(train_dataset, config["NUM_CLIENTS"])

    clients = []
    for i in range(config["NUM_CLIENTS"]):
        is_malicious = i < config["MALICIOUS_CLIENTS"]
        client = Client(i, client_datasets[i], is_malicious)
        clients.append(client)
    print(f"创建了 {config['NUM_CLIENTS']} 个客户端, 其中 {config['MALICIOUS_CLIENTS']} 个是恶意的。")

    # 初始化全局模型
    global_model = SimpleCNN().to(DEVICE)

    # 根据配置选择是否加载已保存的模型
    if config["LOAD_SAVED_MODEL"]:
        print(f"\n--- 跳过训练，直接加载已保存的模型 ---")
        try:
            global_model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location=DEVICE))
            print(f"模型加载成功: {config['MODEL_SAVE_PATH']}")
        except FileNotFoundError:
            print(f"错误: 找不到已保存的模型文件 {config['MODEL_SAVE_PATH']}。请先运行一次完整的训练以生成模型文件。")
            exit()  # 找不到文件则退出程序
    else:

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        global_model_state_dict = global_model.state_dict()
        history_train_loss = []
        history_test_accuracy = []
        history_test_loss = []

        # 创建一个全局优化器和学习率调度器
        # 这个优化器不直接用于训练，仅用于被调度器管理学习率的状态
        global_optimizer = optim.SGD(global_model.parameters(), lr=config["LEARNING_RATE"])

        if config["SCHEDULER_ENABLED"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                global_optimizer,
                mode='min',  # 监控指标越小越好
                factor=config["SCHEDULER_FACTOR"],
                patience=config["SCHEDULER_PATIENCE"],
            )
            print(
                f"\n--- 启用ReduceLROnPlateau学习率调度器: 耐心值={config['SCHEDULER_PATIENCE']}, 衰减因子={config['SCHEDULER_FACTOR']} ---")

        # 初始化早停相关的追踪变量
        if config["EARLY_STOPPING_ENABLED"]:
            patience_counter = 0
            best_performance_metric = float('inf') if config["EARLY_STOPPING_METRIC"] == 'loss' else 0.0
            best_model_state_dict = None
            # print(f"\n--- 启用早停: 耐心值={config['EARLY_STOPPING_PATIENCE']}, 监控指标={config['EARLY_STOPPING_METRIC']} ---")


        # 2. 运行联邦学习
        print(f"\n--- 进行最多 {config["FL_ROUNDS"]} 轮联邦学习以获得一个基础模型 (按 Ctrl+C 可提前中断并保存) ------")
        # 使用 try...except 块包裹整个训练循环
        try:
            for round_num in range(config["FL_ROUNDS"]):
                # (这部分训练循环代码保持不变)
                all_updates = []
                current_round_losses = []
                # 从全局优化器获取当前的学习率
                current_lr = global_optimizer.param_groups[0]['lr']
                for client in clients:
                    update, local_loss = client.train(copy.deepcopy(global_model_state_dict), learning_rate=current_lr)
                    all_updates.append(update)
                    current_round_losses.append(local_loss)
                avg_update = {key: torch.stack([upd[key] for upd in all_updates], dim=0).mean(dim=0) for key in
                              all_updates[0]}
                global_model_state_dict = {key: global_model_state_dict[key] + avg_update[key] for key in
                                           global_model_state_dict}
                global_model.load_state_dict(global_model_state_dict)

                avg_train_loss = np.mean(current_round_losses)
                test_loss, test_accuracy = evaluate_model(global_model, test_loader, DEVICE)
                history_train_loss.append(avg_train_loss)
                history_test_accuracy.append(test_accuracy)
                history_test_loss.append(test_loss)
                print(
                    f"Epoch{round_num + 1}/{config['FL_ROUNDS']} | LR: {current_lr:.6f} | Avg_Train_Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Global_Test_Accuracy: {test_accuracy * 100:.2f}%")

                # 让调度器根据 test_loss 更新学习率
                if config["SCHEDULER_ENABLED"]:
                    scheduler.step(test_loss)

                # 早停判断逻辑
                if config["EARLY_STOPPING_ENABLED"]:

                    # 根据配置选择当前要监控的性能指标
                    current_metric = test_loss if config["EARLY_STOPPING_METRIC"] == 'loss' else test_accuracy

                    # 判断性能是否提升
                    improved = (current_metric < best_performance_metric) if config[
                                                                                 "EARLY_STOPPING_METRIC"] == 'loss' else (
                                current_metric > best_performance_metric)

                    if improved:
                        # print(f"    (性能提升: {best_performance_metric:.4f} -> {current_metric:.4f}. 保存模型并重置耐心.)")
                        best_performance_metric = current_metric
                        patience_counter = 0
                        # 保存当前最好的模型状态
                        best_model_state_dict = copy.deepcopy(global_model.state_dict())
                    else:
                        patience_counter += 1
                        # print(f"    (性能未提升. 耐心计数: {patience_counter}/{config['EARLY_STOPPING_PATIENCE']})")

                    # 如果耐心耗尽，则提前终止训练
                    if patience_counter >= config["EARLY_STOPPING_PATIENCE"]:
                        print(
                            f"\n--- 早停触发: 模型性能已连续 {config['EARLY_STOPPING_PATIENCE']} 轮未提升. 训练在第 {round_num + 1} 轮终止. ---")
                        break  # 跳出训练循环
        except KeyboardInterrupt:
            # 捕获到 Ctrl+C 中断信号后的处理逻辑
            print("\n\n[用户中断] 捕获到 Ctrl+C 信号，正在保存当前模型状态...")
            # 在这里，global_model 保存的是中断前最后一轮完成训练的模型状态
            # 我们将其赋值给 best_model_state_dict，以便后续逻辑统一处理
            best_model_state_dict = copy.deepcopy(global_model.state_dict())
            print("当前模型状态已暂存。")


        print("\n基础模型训练完成或被中断。")

        # 如果早停被触发，加载并保存历史上最好的模型
        if best_model_state_dict is not None:
            global_model.load_state_dict(best_model_state_dict)
            print("已加载性能最佳的模型状态用于后续分析。")

        # 确保保存模型的目录存在
        save_dir = os.path.dirname(config["MODEL_SAVE_PATH"])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(global_model.state_dict(), config["MODEL_SAVE_PATH"])
        print(f"\n最佳模型已保存至: {config['MODEL_SAVE_PATH']}")


    # 无论是否训练，都获取最终的global_model_state_dict以进行后续分析
    final_global_model_state_dict = global_model.state_dict()

    # --- 4. 关键验证步骤：实现“不稳定增量”分析 ---
    print(f"\n--- 关键验证：使用 {config['NUM_PROBES']} 个探测器进行“不稳定增量”分析 ---")
    print(f"扰动强度：{config['PERTURBATION_STRENGTH']}")

    # --- 准备工作 (与之前类似) ---
    if config["LOAD_SAVED_MODEL"]:
        current_lr = config["LEARNING_RATE"]
    else:
        current_lr = global_optimizer.param_groups[0]['lr']

    malicious_client = clients[0]
    benign_client = clients[-1]

    # 生成恶意和良性更新
    malicious_update, _ = malicious_client.train(copy.deepcopy(final_global_model_state_dict),
                                                 learning_rate=current_lr)
    benign_update, _ = benign_client.train(copy.deepcopy(final_global_model_state_dict), learning_rate=current_lr)

    # --- 核心修改：实现不稳定增量分析 ---
    if config["CHANGE_SEED"]:
        current_time_seed = int(time.time())
        np.random.seed(current_time_seed)
    else:
        np.random.seed(config["SEED"])

    probe_images = []
    for _ in range(config["NUM_PROBES"]):
        random_image_index = np.random.randint(len(test_dataset))
        probe_image_sample, _ = test_dataset[random_image_index]
        probe_images.append(probe_image_sample)
    if config["CHANGE_SEED"]:
        print(f"已生成一组随机的 {config['NUM_PROBES']} 张探测图片。")
    else:
        print(f"已生成一组固定的 {config['NUM_PROBES']} 张探测图片。")

    # 2. 计算基线不稳定性得分 (score_base)
    # 这是未被任何客户端更新修改的、原始全局模型的不稳定性
    base_scores = []
    for probe_image in probe_images:
        # 注意：这里的 client_update 参数我们传入一个空的“假更新”，因为它不会被使用
        # 一个小技巧：创建一个和模型参数同类型但全为0的张量作为假更新
        dummy_update = {key: torch.zeros_like(param) for key, param in final_global_model_state_dict.items()}
        # 在分析函数内部，temp_model = global_model + dummy_update，所以temp_model就是global_model
        score = internal_consistency_analysis(copy.deepcopy(final_global_model_state_dict), dummy_update, DEVICE,
                                              probe_image)
        base_scores.append(score)

    score_base = np.max(base_scores)  # 同样使用max聚合
    print(f"全局模型的基线不稳定性得分 (Score_Base): {score_base:.6f}")

    # 3. 计算应用更新后的临时模型得分 (score_temp) 并计算增量
    # (良性客户端)
    benign_temp_scores = []
    for probe_image in probe_images:
        score = internal_consistency_analysis(copy.deepcopy(final_global_model_state_dict), benign_update, DEVICE,
                                              probe_image)
        benign_temp_scores.append(score)
    score_benign_temp = np.max(benign_temp_scores)
    benign_delta = score_benign_temp - score_base  # 计算增量

    # (恶意客户端)
    malicious_temp_scores = []
    for probe_image in probe_images:
        score = internal_consistency_analysis(copy.deepcopy(final_global_model_state_dict), malicious_update, DEVICE,
                                              probe_image)
        malicious_temp_scores.append(score)
    score_malicious_temp = np.max(malicious_temp_scores)
    malicious_delta = score_malicious_temp - score_base  # 计算增量

    # --- 5. 打印基于“增量”的结果 ---
    print("\n--- 实验结果 (不稳定增量分析) ---")
    print(f"良性更新引入的“不稳定增量”: {benign_delta:.6f} ({score_benign_temp:.6f})")
    print(f"恶意更新引入的“不稳定增量”: {malicious_delta:.6f} ({score_malicious_temp:.6f})")

    # --- 最终版判断逻辑 (包含更智能的打印) ---
    conclusion = "[结论]: 实验结果不显著。"
    signal_type = "无显著信号"  # 用于描述信号类型
    ratio = None

    # 只要良性增量不为零，就计算比率
    if benign_delta != 0:
        ratio = malicious_delta / benign_delta
        print(f"恶意/良性增量比率: {ratio:.6f}")  # 仍然打印原始比率，作为核心数据点

    else:
        # 特殊处理良性增量恰好为零的情况
        ratio = float('inf') if malicious_delta > 0 else 0
        print(f"恶意/良性增量比率: {ratio} (良性增量为零)")

    # 定义成功的三个主要场景：
    # 场景1: “经典成功” -> 恶意更新是更差的“破坏者”
    if benign_delta >= 0 and malicious_delta > benign_delta * 1.5:

        signal_type = "恶意更新是更强的破坏者 (Case 1)"
        conclusion = "[结论]: 实验成功！"

    # 场景2: “理想成功” -> 良性更新在稳定模型，而恶意更新在破坏模型
    elif benign_delta < 0 < malicious_delta:
        signal_type = "良性稳定 vs. 恶意破坏 (Case 2)"
        conclusion = "[结论]: 实验成功！"

    # 场景3: “新发现的成功” -> 良性更新是更强的“稳定器”
    elif benign_delta < 0 and malicious_delta < 0 and ratio is not None and ratio < 0.7:
        signal_type = "良性是更强的稳定器 (Case 3)"
        conclusion = "[结论]: 实验成功！"

    # 特殊情况的判断
    elif benign_delta == 0 and malicious_delta > 0.0001:
        signal_type = "良性无影响 vs. 恶意破坏 (Special Case)"
        conclusion = "[结论]: 实验成功！"

    # 打印出识别到的信号类型
    print(f"识别到的信号类型: {signal_type}")
    print("case1:比率越大越好；case2:比率的绝对值越大越好；case3:比率越小越好")
    print(f"\n{conclusion}")

    if not config["LOAD_SAVED_MODEL"]:
        # 获取实际运行的轮数
        num_actual_rounds = len(history_train_loss)
        x_axis = range(1, num_actual_rounds + 1)

        # 绘制结果图表
        plt.figure(figsize=(12, 5))

        # 第一个子图为训练与测试损失的对比图
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, history_train_loss, marker='o', label='Train Loss')  # 为曲线添加标签
        plt.plot(x_axis, history_test_loss, marker='x', label='Test Loss')  # 新增：绘制测试损失曲线
        plt.title("Training vs. Test Loss per Round")  # 修改标题
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Loss")
        plt.legend()  # 显示图例
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # 修改前: plt.plot(range(1, config["FL_ROUNDS"] + 1), history_test_accuracy, marker='s', color='r')
        plt.plot(x_axis, history_test_accuracy, marker='s', color='r')  # 使用新的x_axis
        plt.title("Global Model Test Accuracy per Round")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
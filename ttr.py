# -*- coding: utf-8 -*-
"""
@File    : ttr.py
@Time    : 2025/9/25
@Author  : dizjia
@Description: 成对余弦相似度+IDA双重审查
            没有wandb版本
@History :
- 2025/9/25, v1.2：
    - 将IDA反转使用，成对余弦相似度保留
- 2025/9/25, v1.1：
    - 成对余弦相似度+IDA双重审查
    - 在后门攻击场景下效果依然很差
"""


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
import csv
import argparse



# --- 0. 全局配置字典 ---
config = {
    # 实验环境设置
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "mps"),
    "SEED": 123,

    # 实验日志文件路径
    "LOG_FILE_PATH": "./log/ttr_simulation/scene3.csv",

    # 联邦学习设置
    "NUM_CLIENTS": 20,  # 增加客户端总数以更好地模拟统计检测
    "FL_ROUNDS": 200,  # 模拟的总轮数
    "CLIENTS_PER_ROUND": 10,  # 每轮随机挑选10个客户端参与

    # 客户端本地训练参数
    "LOCAL_EPOCHS": 3,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.01,

    # 攻击设置
    "ATTACK_TYPE": "backdoor",
    "MALICIOUS_CLIENTS": 8,
    "POISON_RATIO": 0.7,
    "BACKDOOR_TRIGGER_SIZE": 5,
    "BACKDOOR_TARGET_LABEL": 0,

    # 防御设置
    "DEFENSE_ENABLED": True,  # 是否启用在线防御
    "DEFENSE_START_ROUND": 5,  # 从第5轮开始执行防御，给模型一点初始收敛时间

    # 双重审查模型参数
    "PERTURBATION_STRENGTH": 0.1,  # 扰动强度
    "SIMILARITY_LOW_MAD_THRESHOLD": 2.0, # 相似度声望分的“低分”阈值 (越小越宽松)
    "IDA_LOW_MAD_THRESHOLD": 2.0,         # IDA不稳定性增量的“低分”阈值 (越小越宽松)

    "CLIP_MAX_NORM":1.2,  #裁剪阈值

    # 声誉与降权模块参数

    "REPUTATION_DECAY_FACTOR": 0.8, # γ值 (gamma)


    # 学习率调度器 (Scheduler) 设置
    "SCHEDULER_ENABLED": True,
    "SCHEDULER_PATIENCE": 5,   # 连续5轮 test_loss 不下降就调整
    "SCHEDULER_FACTOR": 0.5,   # 学习率衰减为原来的一半 (lr = lr * 0.5)

    # 是否早停
    "EARLY_STOPPING_ENABLED": True,      # True启用早停
    "EARLY_STOPPING_PATIENCE": 10,       # 连续10轮性能不提升就停止
    "EARLY_STOPPING_METRIC": "loss",     # 监控指标: 'loss' 或 'accuracy'

}


def log_experiment_results(log_file, scenario_name, config_params, metrics):
    """
    将单次实验场景的配置和结果记录到CSV文件中。
    """
    # 将场景名、配置和结果合并到一个字典中
    log_data = {"scenario": scenario_name, **config_params, **metrics}

    # 定义表头顺序 (动态生成以包含所有可能的键)
    header = ["scenario"] + list(config_params.keys()) + list(metrics.keys())

    # 检查文件是否存在
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()

            writer.writerow(log_data)

    except IOError as e:
        print(f"[错误] 无法写入日志文件: {log_file}, 原因: {e}")


# --- 1. 模型与数据处理 (与之前版本基本一致) ---
# ... (SimpleCNN, get_cifar10_data, create_iid_partitions 等函数定义在此处，无需修改) ...
# 为了简洁，此处省略，请您从旧文件复制过来
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_stack = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_stack = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 512), nn.ReLU())
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        return self.classifier(self.fc_stack(self.conv_stack(x)))

    def get_penultimate_features(self, x):
        return self.fc_stack(self.conv_stack(x))


def get_cifar10_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def create_iid_partitions(dataset, num_clients):
    num_items = len(dataset) // num_clients
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return [Subset(dataset, list(idxs)) for _, idxs in dict_users.items()]


class BackdoorDataset(Dataset):
    """
    一个为数据集植入后门攻击的包装器。
    它会在一部分图片上粘贴一个触发器（例如，右下角的白色方块），
    并将其标签修改为指定的目标标签。
    """

    def __init__(self, original_dataset, poison_ratio, trigger_size, target_label):
        self.original_dataset = original_dataset
        self.poison_ratio = poison_ratio
        self.trigger_size = trigger_size
        self.target_label = target_label

        # 随机选择要投毒的样本索引
        self.poison_indices = set(np.random.choice(
            len(self), int(len(self) * self.poison_ratio), replace=False
        ))

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始图片和标签
        img, label = self.original_dataset[idx]

        # 如果当前样本需要被投毒
        if idx in self.poison_indices:
            # 1. 修改标签
            label = self.target_label

            # 2. 在图片上粘贴触发器
            # 我们将触发器放在右下角
            # img 的形状是 (C, H, W)，例如 (3, 32, 32)
            c, h, w = img.shape
            # 将右下角的一个 trigger_size x trigger_size 的区域像素值设为最大（白色）
            img[:, h - self.trigger_size:, w - self.trigger_size:] = 1

        return img, label


def evaluate_backdoor_asr(model, test_loader, device, config):
    """
    计算后门攻击的攻击成功率 (ASR)。
    """
    model.eval()
    success, total = 0, 0

    trigger_size = config["BACKDOOR_TRIGGER_SIZE"]
    target_label = config["BACKDOOR_TARGET_LABEL"]

    with torch.no_grad():
        for data, target in test_loader:
            # 筛选出那些不是目标标签的图片
            non_target_mask = (target != target_label)
            data, target = data[non_target_mask], target[non_target_mask]

            if data.size(0) == 0: continue

            data = data.to(device)
            total += data.size(0)

            # 在图片上粘贴触发器
            c, h, w = data.shape[1:]
            data[:, :, h - trigger_size:, w - trigger_size:] = 1

            # 进行预测
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            # 如果预测结果等于目标标签，则攻击成功
            success += (predicted == target_label).sum().item()

    return (success / total) * 100 if total > 0 else 0


class LabelFlippingDataset(Dataset):
    def __init__(self, original_dataset, poison_ratio, num_classes=10):
        self.original_dataset, self.poison_ratio, self.num_classes = original_dataset, poison_ratio, num_classes
        self.poison_indices = set(np.random.choice(len(self), int(len(self) * self.poison_ratio), replace=False))

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        if idx in self.poison_indices:
            possible_wrong_labels = list(range(self.num_classes))
            possible_wrong_labels.remove(label)
            return img, random.choice(possible_wrong_labels)
        return img, label


class Client:
    def __init__(self, client_id, local_data, is_malicious=False):
        self.client_id, self.local_data, self.is_malicious = client_id, local_data, is_malicious
        self.device = config["DEVICE"]
        if is_malicious:
            # [核心修正] 根据ATTACK_TYPE来选择攻击方式
            if config["ATTACK_TYPE"] == "label_flipping":
                self.local_data = LabelFlippingDataset(local_data, poison_ratio=config["POISON_RATIO"])
            elif config["ATTACK_TYPE"] == "backdoor":
                # (确保您代码中已经定义了BackdoorDataset类)
                self.local_data = BackdoorDataset(
                    original_dataset=local_data,
                    poison_ratio=config["POISON_RATIO"],
                    trigger_size=config["BACKDOOR_TRIGGER_SIZE"],
                    target_label=config["BACKDOOR_TARGET_LABEL"]
                )
        self.dataloader = DataLoader(self.local_data, batch_size=config["BATCH_SIZE"], shuffle=True)

    def train(self, global_model_state_dict, learning_rate):
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # [修改] 差异化本地训练轮数
        epochs_to_run = config["LOCAL_EPOCHS"]
        if self.is_malicious:
            epochs_to_run = config["LOCAL_EPOCHS"] * 4  # 例如，恶意客户端的训练轮数是诚实的2倍
            # 您也可以在这里设置一个固定的更多轮数，比如 10

        model.train()
        for _ in range(epochs_to_run):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(data), target)
                loss.backward()
                optimizer.step()
        return {key: model.state_dict()[key] - global_model_state_dict[key] for key in global_model_state_dict}


def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # 计算损失
            loss = nn.CrossEntropyLoss()(outputs, target)
            total_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / total  # 计算平均损失
    accuracy = correct / total
    return avg_loss, accuracy  # 返回两个值


# --- 2. 核心防御逻辑与服务器定义 ---

# [修改] 新的、更模块化的辅助函数
def calculate_instability_score(model_state_dict, probe_images, device, config):
    """
    一个通用的函数，用于计算给定模型状态的不稳定性分数。
    Args:
        model_state_dict (dict): 模型的state_dict。
        probe_images (list): 用于测试的探针图片列表。
        device: 计算设备。
        config (dict): 全局配置字典。

    Returns:
        float: 模型的不稳定性分数（平均余弦不相似度）。
    """
    model = SimpleCNN().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    scores = []
    with torch.no_grad():
        for img in probe_images:
            probe_X = img.unsqueeze(0).to(device)
            perturbation = torch.randn_like(probe_X) * config["PERTURBATION_STRENGTH"]
            probe_X_prime = probe_X + perturbation
            features_X = model.get_penultimate_features(probe_X)
            features_X_prime = model.get_penultimate_features(probe_X_prime)
            scores.append(1 - nn.functional.cosine_similarity(features_X, features_X_prime).item())

    return np.mean(scores)

def clip_update_norm_(update, max_norm):
    """
    对单个客户端的更新字典 (update) 进行原地范数裁剪。
    Args:
        update (dict): 客户端模型更新的 state_dict。
        max_norm (float): 范数上限。
    """
    flat_update = torch.cat([p.flatten() for p in update.values()])
    norm = torch.norm(flat_update, p=2)  # p=2指明使用L2范数
    if norm > max_norm:
        clip_coef = max_norm / (norm + 1e-6)
        for param in update.values():
            param.mul_(clip_coef)

class Server:
    def __init__(self, all_clients, test_loader, config):
        """
        服务器初始化
        Args:
            all_clients (list): 包含所有客户端实例的列表。
            test_loader (DataLoader): 用于全局模型评估的测试数据加载器。
            config (dict): 当前场景的配置字典。
        """
        self.device = config["DEVICE"]
        self.config = config  # 保存当前场景的配置
        self.global_model = SimpleCNN().to(self.device)
        self.test_loader = test_loader

        # 客户端管理
        self.all_clients = all_clients
        self.active_clients_pool = list(all_clients)
        self.reputation_scores = {client.client_id: 0 for client in all_clients}

        # --- [新增] 创建一个恶意ID集合，方便快速查询 ---
        self.malicious_ids = {c.client_id for c in all_clients if c.is_malicious}

        # 探测器图片生成
        self.probe_images = self._generate_probes()

        # 历史记录
        self.history = {'accuracy': [], 'loss': [], 'removed_malicious': [], 'removed_benign': []}

        # 在服务器内部管理优化器、调度器和早停状态
        self.optimizer = optim.SGD(self.global_model.parameters(), lr=self.config["LEARNING_RATE"])

        # 初始化学习率调度器
        self.scheduler = None
        if self.config["SCHEDULER_ENABLED"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config["SCHEDULER_FACTOR"],
                patience=self.config["SCHEDULER_PATIENCE"],
            )
            # print(f"    [服务器设置] 学习率调度器已启用。")

        # 初始化早停变量
        if self.config["EARLY_STOPPING_ENABLED"]:
            self.patience_counter = 0
            self.best_performance_metric = float('inf') if self.config["EARLY_STOPPING_METRIC"] == 'loss' else 0.0
            self.best_model_state_dict = None
            # print(f"    [服务器设置] 早停策略已启用。")

    def _generate_probes(self):
        # [修改] 从预先计算好的文件中加载基于特征聚类的探针图片索引
        probe_indices_file = "ida_simulation/clustered_probe_indices_for_cifar10.npy"
        try:
            # 加载预先计算好的索引
            indices = np.load(probe_indices_file)
            print(f"    [服务器设置] 成功从 {probe_indices_file} 加载了 {len(indices)} 个探针。")
        except FileNotFoundError:
            print(f"    [错误] 探针索引文件 {probe_indices_file} 未找到！")
            print("    [提示] 请先运行 create_probes.py 脚本来生成该文件。")
            # 也可以在这里提供一个备用方案，比如退回随机抽样，或者直接退出
            raise

        probe_images = []
        for idx in indices:
            # np.load 可能会返回 numpy.int64，需要转换为Python int
            img, _ = self.test_loader.dataset[int(idx)]
            probe_images.append(img)
        return probe_images


    def run_simulation(self):
        """运行完整的联邦学习模拟过程 (v3.3 - 永久声誉 + 指数衰减权重)"""
        # --- [新增] 用于收集范数的列表 ---
        benign_norms = []
        malicious_norms = []

        try:
            for t in range(self.config["FL_ROUNDS"]):
                print(f"\n--- 第 {t + 1}/{self.config['FL_ROUNDS']} 轮 ---")

                num_to_select = min(self.config["CLIENTS_PER_ROUND"], len(self.active_clients_pool))
                selected_clients = random.sample(self.active_clients_pool, num_to_select)

                # --- [新增代码] 打印本轮选中的恶意客户端ID ---
                # 检查当前是否为场景二或三
                if self.config.get("MALICIOUS_CLIENTS", 0) > 0:
                    # 从选中的客户端中筛选出恶意的，并记录他们的ID
                    selected_malicious_ids = sorted([
                        client.client_id for client in selected_clients if client.is_malicious
                    ])
                    print(
                        f"    [攻击监控] 本轮选中了 {len(selected_malicious_ids)} 个恶意客户端: {selected_malicious_ids}")

                client_updates = {}
                global_model_state_dict = self.global_model.state_dict()
                current_lr = self.optimizer.param_groups[0]['lr']

                for client in selected_clients:
                    update = client.train(copy.deepcopy(global_model_state_dict), learning_rate=current_lr)
                    client_updates[client.client_id] = update

                # --- [新增] 在防御逻辑开始前，计算并收集范数 ---
                for client_id, update in client_updates.items():
                    flat_update = torch.cat([p.flatten() for p in update.values()])
                    norm = torch.norm(flat_update, p=2).item()  # .item() 将tensor转为数字

                    if client_id in self.malicious_ids:
                        malicious_norms.append(norm)
                    else:
                        benign_norms.append(norm)

                if self.config["DEFENSE_ENABLED"] and t + 1 >= self.config["DEFENSE_START_ROUND"]:

                    # --- “双重审查”防御模型 ---
                    # --- Part 0: 预处理 (梯度裁剪) ---
                    CLIP_MAX_NORM = config["CLIP_MAX_NORM"] # 这是一个超参数，您可以按需调整
                    # 注意：我们克隆一份用于裁剪，以防未来需要原始更新
                    clipped_updates = copy.deepcopy(client_updates)
                    for client_id in clipped_updates.keys():
                        clip_update_norm_(clipped_updates[client_id], CLIP_MAX_NORM)
                    # print(f"    [防御流程] 已对更新执行范数裁剪 (上限={CLIP_MAX_NORM})，用于后续分析。")
                    client_ids = list(clipped_updates.keys())
                    num_clients_in_round = len(client_ids)

                    # --- Part 1: 行为异常检测 (基于两两相似度) ---
                    behavioral_suspects = []
                    if num_clients_in_round > 1:
                        updates_flat = {cid: torch.cat([p.flatten() for p in upd.values()]) for cid, upd in
                                        clipped_updates.items()}
                        similarity_matrix = np.ones((num_clients_in_round, num_clients_in_round))
                        for i in range(num_clients_in_round):
                            for j in range(i + 1, num_clients_in_round):
                                client_i_id, client_j_id = client_ids[i], client_ids[j]
                                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                                similarity = cos(updates_flat[client_i_id], updates_flat[client_j_id]).item()
                                similarity_matrix[i, j] = similarity
                                similarity_matrix[j, i] = similarity
                        reputation_scores_round = np.sum(similarity_matrix, axis=1)

                        median_rep = np.median(reputation_scores_round)
                        mad_rep = np.median(np.abs(reputation_scores_round - median_rep))
                        if mad_rep == 0: mad_rep = 1e-9

                        score_threshold_sim = self.config["SIMILARITY_LOW_MAD_THRESHOLD"]
                        for i in range(num_clients_in_round):
                            score = reputation_scores_round[i]
                            z_score = (score - median_rep) / mad_rep
                            if z_score < -score_threshold_sim:
                                behavioral_suspects.append(client_ids[i])

                    print(f"    [调试-行为] 检测到低相似度嫌疑: {sorted(behavioral_suspects)}")

                    # --- Part 2: 后果异常检测 (基于“反转”的IDA) ---
                    consequential_suspects = []
                    score_base = calculate_instability_score(global_model_state_dict, self.probe_images, self.device,
                                                             self.config)
                    deltas = {cid: calculate_instability_score({k: global_model_state_dict[k] + u[k] for k in u},
                                                               self.probe_images, self.device, self.config) - score_base
                              for cid, u in client_updates.items()}

                    delta_values = list(deltas.values())
                    if len(delta_values) > 1:
                        median_delta = np.median(delta_values)
                        mad_delta = np.median(np.abs(delta_values - median_delta))
                        if mad_delta == 0: mad_delta = 1e-9

                        score_threshold_ida = self.config["IDA_LOW_MAD_THRESHOLD"]
                        for client_id, delta in deltas.items():
                            score_mad = (delta - median_delta) / mad_delta
                            if score_mad < -score_threshold_ida:
                                consequential_suspects.append(client_id)

                    print(f"    [调试-后果] 检测到低稳定性嫌疑: {sorted(consequential_suspects)}")

                    # --- Part 3: 取交集，最终裁决 (AND Logic) ---
                    suspicious_ids = sorted(list(set(behavioral_suspects) & set(consequential_suspects)))

                    if suspicious_ids:
                        print(f"    [侦测模块] 本轮最终可疑客户端: {suspicious_ids}")

                    # 后续的声誉累积和加权聚合逻辑保持不变
                    for client_id in suspicious_ids:
                        self.reputation_scores[client_id] += 1

                    decay_factor = self.config["REPUTATION_DECAY_FACTOR"]
                    weights = {cid: decay_factor ** self.reputation_scores.get(cid, 0) for cid in
                               clipped_updates.keys()}
                    sum_of_weights = sum(weights.values())
                    avg_update = {}
                    for key in clipped_updates[list(clipped_updates.keys())[0]].keys():
                        weighted_sum_layer = torch.stack(
                            [clipped_updates[cid][key] * weights[cid] for cid in clipped_updates.keys()],
                            dim=0
                        ).sum(dim=0)
                        avg_update[key] = weighted_sum_layer / sum_of_weights
                else:  # 如果不启用防御，则走标准FedAvg
                    updates_to_aggregate = client_updates
                    if updates_to_aggregate:
                        avg_update = {
                            key: torch.stack([upd[key] for upd in updates_to_aggregate.values()], dim=0).mean(dim=0)
                            for key in updates_to_aggregate[list(updates_to_aggregate.keys())[0]]
                        }
                    else:
                        avg_update = None  # 如果没有可聚合的更新

                # --- 5. 模型应用更新 ---
                if avg_update:
                    with torch.no_grad():
                        for param_name, param in self.global_model.named_parameters():
                            param += avg_update[param_name]

                # --- 6. 评估与记录 ---
                test_loss, accuracy = evaluate_model(self.global_model, self.test_loader, self.device)
                self.history['accuracy'].append(accuracy)
                self.history['loss'].append(test_loss)

                print(
                    f"    >> 第 {t + 1} 轮结束 | 当前LR: {current_lr:.6f} | 测试损失: {test_loss:.4f} | 全局模型准确率: {accuracy * 100:.2f}%")

                # --- 7. 调度器与早停 ---
                if self.scheduler:
                    self.scheduler.step(test_loss)

                if self.config["EARLY_STOPPING_ENABLED"]:
                    current_metric = test_loss if self.config["EARLY_STOPPING_METRIC"] == 'loss' else accuracy
                    is_loss = self.config["EARLY_STOPPING_METRIC"] == 'loss'
                    improved = (current_metric < self.best_performance_metric) if is_loss else (
                                current_metric > self.best_performance_metric)

                    if improved:
                        self.best_performance_metric = current_metric
                        self.patience_counter = 0
                        self.best_model_state_dict = copy.deepcopy(self.global_model.state_dict())
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.config["EARLY_STOPPING_PATIENCE"]:
                        print(f"    !! 早停触发: 训练在第 {t + 1} 轮终止 !!")
                        break

        except KeyboardInterrupt:
            print("\n\n[用户中断] 已捕获 Ctrl+C 信号...")
            self.best_model_state_dict = copy.deepcopy(self.global_model.state_dict())

        # ... (函数剩余部分，如加载最佳模型、保存模型等，保持不变) ...
        if self.config["EARLY_STOPPING_ENABLED"] and self.best_model_state_dict is not None:
            self.global_model.load_state_dict(self.best_model_state_dict)
            # print("\n已加载性能最佳的模型状态。")

        scenario_name = self.config.get("SCENARIO_NAME", "model")
        save_path = f"./saved_models/{scenario_name}_final.pth"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.global_model.state_dict(), save_path)
        # print(f"当前场景的最终模型已保存至: {save_path}")

        # --- [新增] 在模拟结束后，打印统计数据 ---
        print("\n--- 范数侦察统计 ---")
        if benign_norms:
            print(f"良性更新范数 (Benign Norms) - 共 {len(benign_norms)} 个样本:")
            print(f"  - 均值 (Mean): {np.mean(benign_norms):.4f}")
            print(f"  - 中位数 (Median): {np.median(benign_norms):.4f}")
            print(f"  - 95百分位 (95th percentile): {np.percentile(benign_norms, 95):.4f}")
            print(f"  - 最大值 (Max): {np.max(benign_norms):.4f}")
        if malicious_norms:
            print(f"恶意更新范数 (Malicious Norms) - 共 {len(malicious_norms)} 个样本:")
            print(f"  - 均值 (Mean): {np.mean(malicious_norms):.4f}")
            print(f"  - 中位数 (Median): {np.median(malicious_norms):.4f}")
            print(f"  - 最小值 (Min): {np.min(malicious_norms):.4f}")

        return self.history


# --- 3. 实验运行与对比 ---
if __name__ == "__main__":
    # [新增] 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="选择性运行联邦学习模拟场景。")
    parser.add_argument(
        '--scenarios','-s',
        nargs='+',  # 允许多个值
        type=int,
        default=[1, 2, 3],  # 如果不指定，则默认运行所有场景
        choices=[1, 2, 3],
        help="指定要运行的场景编号列表。1:无攻击, 2:有攻击无防御, 3:有攻击有防御。例如: --scenarios 1 3或-s 1 3"
    )
    args = parser.parse_args()
    scenarios_to_run = args.scenarios
    # print(f"[实验配置] IDA阈值 = {config["OUTLIER_THRESHOLD"]},相似度阈值 = {config["SIMILARITY_THRESHOLD"]},声誉容忍度 = {config["REPUTATION_THRESHOLD"]},扰动强度 = {config["PERTURBATION_STRENGTH"]}")

    set_seed(config["SEED"])

    # 1. 准备全局数据和客户端池
    train_dataset, test_dataset = get_cifar10_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_datasets = create_iid_partitions(train_dataset, config["NUM_CLIENTS"])

    # 定义需要记录到日志的通用参数
    common_params_to_log = {
        "LOCAL_EPOCHS": config["LOCAL_EPOCHS"],
        "LEARNING_RATE": config["LEARNING_RATE"],
    }

    # [修改] 2. 初始化所有历史记录变量，以防某些场景被跳过
    history_no_attack = None
    history_under_attack = None
    history_with_defense = None

    # [修改] 3. 使用 "if" 语句包裹每个场景
    if 1 in scenarios_to_run:

        # --- 场景一: 无攻击 ---
        print("\n\n=============== 场景一: 无攻击环境 ===============")

        config_no_attack = copy.deepcopy(config)
        config_no_attack["MALICIOUS_CLIENTS"] = 0
        config_no_attack["DEFENSE_ENABLED"] = False
        config_no_attack["SCENARIO_NAME"] = "no_attack"

        clients_no_attack = [Client(i, client_datasets[i], is_malicious=False) for i in range(config["NUM_CLIENTS"])]
        # 将更新后的config传入Server
        server_no_attack = Server(clients_no_attack, test_loader, config_no_attack)
        history_no_attack = server_no_attack.run_simulation()


        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_no_attack.global_model, test_loader, config["DEVICE"],
                                          config_no_attack)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_1 = common_params_to_log.copy()
        # 场景一没有攻击和防御，用 N/A 覆盖
        params_log_1.update({
            "MALICIOUS_CLIENTS":"N/A",
            "POISON_RATIO": "N/A",
            "PERTURBATION_STRENGTH": "N/A",
            "SIMILARITY_LOW_MAD_THRESHOLD": "N/A",
            "IDA_LOW_MAD_THRESHOLD": "N/A",

            "REPUTATION_DECAY_FACTOR": "N/A"
        })
        actual_rounds_1 = len(history_no_attack['accuracy'])
        metrics_log_1 = {
            "actual_rounds": actual_rounds_1,
            "final_accuracy": f"{history_no_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A",
            "final_asr": f"{final_asr:.2f}%",
        }
        log_experiment_results(config["LOG_FILE_PATH"], "1", params_log_1, metrics_log_1)



    if 2 in scenarios_to_run:
        # --- 场景二: 有攻击, 无防御 ---
        print("\n\n=============== 场景二: 有攻击, 无防御 ===============")

        config_under_attack = copy.deepcopy(config)
        config_under_attack["DEFENSE_ENABLED"] = False
        config_under_attack["SCENARIO_NAME"] = "under_attack_no_defense"

        clients_under_attack = [Client(i, client_datasets[i], is_malicious=(i < config_under_attack["MALICIOUS_CLIENTS"]))
                                for i in range(config["NUM_CLIENTS"])]
        # [新增] 打印本场景的恶意客户端ID列表
        malicious_ids_2 = [c.client_id for c in clients_under_attack if c.is_malicious]
        print(f"    [实验设置] 本场景恶意客户端ID: {malicious_ids_2}")

        server_under_attack = Server(clients_under_attack, test_loader, config_under_attack)
        history_under_attack = server_under_attack.run_simulation()


        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_under_attack.global_model, test_loader, config["DEVICE"],
                                          config_under_attack)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_2 = common_params_to_log.copy()
        params_log_2.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": "N/A",
            "SIMILARITY_LOW_MAD_THRESHOLD": "N/A",
            "IDA_LOW_MAD_THRESHOLD": "N/A",

            "REPUTATION_DECAY_FACTOR": "N/A",
        })
        actual_rounds_2 = len(history_under_attack['accuracy'])
        metrics_log_2 = {
            "actual_rounds": actual_rounds_2,
            "final_accuracy": f"{history_under_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A",
            "final_asr": f"{final_asr:.2f}%"
        }
        log_experiment_results(config["LOG_FILE_PATH"], "2", params_log_2, metrics_log_2)


    if 3 in scenarios_to_run:
        # --- 场景三: 有攻击, 有您的IDA防御 ---
        print("\n\n=============== 场景三: 有攻击, 启用IDA防御 ===============")
        print(f"    [攻击设置] {config["ATTACK_TYPE"]}")

        config_with_defense = copy.deepcopy(config)
        config_with_defense["DEFENSE_ENABLED"] = True
        config_with_defense["SCENARIO_NAME"] = "with_defense"

        clients_with_defense = [Client(i, client_datasets[i], is_malicious=(i < config_with_defense["MALICIOUS_CLIENTS"]))
                                for i in range(config["NUM_CLIENTS"])]

        # [新增] 打印本场景的恶意客户端ID列表
        malicious_ids_3 = [c.client_id for c in clients_with_defense if c.is_malicious]
        print(f"    [实验设置] 本场景恶意客户端ID: {malicious_ids_3}")

        server_with_defense = Server(clients_with_defense, test_loader, config_with_defense)
        history_with_defense = server_with_defense.run_simulation()


        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_with_defense.global_model, test_loader, config["DEVICE"],
                                          config_with_defense)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_3 = common_params_to_log.copy()
        params_log_3.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": config["PERTURBATION_STRENGTH"],
            "SIMILARITY_LOW_MAD_THRESHOLD": config["SIMILARITY_LOW_MAD_THRESHOLD"],
            "IDA_LOW_MAD_THRESHOLD": config.get("IDA_LOW_MAD_THRESHOLD", "N/A"),

            "REPUTATION_DECAY_FACTOR": config.get("REPUTATION_DECAY_FACTOR", "N/A")
        })
        actual_rounds_3 = len(history_with_defense['accuracy'])
        # [修改] 2. 正确地统计并记录最终被“降权”的客户端数量
        # 对于“指数衰减”策略，我们定义“被降权”为最终声誉分 > 0 的客户端
        final_reputations = server_with_defense.reputation_scores
        convicted_clients = {cid for cid, rep in final_reputations.items() if rep > 0}

        malicious_clients_config = config_with_defense["MALICIOUS_CLIENTS"]
        downweighted_malicious_count = len([cid for cid in convicted_clients if cid < malicious_clients_config])
        downweighted_benign_count = len([cid for cid in convicted_clients if cid >= malicious_clients_config])

        metrics_log_3 = {
            "actual_rounds": actual_rounds_3,
            "final_accuracy": f"{history_with_defense['accuracy'][-1] * 100:.2f}%",
            "downweighted_malicious": downweighted_malicious_count,
            "downweighted_benign": downweighted_benign_count,
            "final_asr": f"{final_asr:.2f}%"
        }
        log_experiment_results(config["LOG_FILE_PATH"], "3", params_log_3, metrics_log_3)

    # print(f"\n[日志] 所有场景已成功记录到: {config['LOG_FILE_PATH']}")

    if history_with_defense:
        print("\n--- 防御效果总结 ---")
        final_reputations = server_with_defense.reputation_scores
        suspicious_clients_sorted = sorted(
            [(cid, rep) for cid, rep in final_reputations.items() if rep > 0],
            key=lambda item: item[1],
            reverse=True
        )

        print("最终声誉分 > 0 的客户端 (从高到低):")
        if not suspicious_clients_sorted:
            print("    没有任何客户端的最终声誉分 > 0。")
        else:
            for cid, rep in suspicious_clients_sorted:
                is_malicious_str = "恶意" if cid < config_with_defense["MALICIOUS_CLIENTS"] else "良性"
                print(f"    客户端 {cid} ({is_malicious_str}): 声誉分 = {rep}")


    # --- 4. 结果可视化 ---
    # [修改] 将两个图的绘制逻辑分得更清晰

    # --- 图表一：准确率对比 ---
    plt.figure(figsize=(10, 6))
    max_rounds_acc = 0

    if history_no_attack:
        rounds = len(history_no_attack['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_no_attack['accuracy'], 'g-s',
                 label=f'No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_under_attack['accuracy'], 'r-x',
                 label=f'Under Attack (Ran for {rounds} rounds)')

    # [修正] 补上缺失的 "With Defense" 准确率曲线
    if history_with_defense:
        rounds = len(history_with_defense['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_with_defense['accuracy'], 'b-o',
                 label=f'With IDA Defense (Converged in {rounds} rounds)')

    if max_rounds_acc > 0:
        plt.title("Performance Comparison of IDA Online Defense (Accuracy)")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlim(0, max_rounds_acc + 5)
        plt.show()
    else:
        # 如果一个场景都没跑，就打印提示
        print("\n[可视化] 未运行任何场景，无法绘制准确率图。")

    # --- 图表二：损失值对比 ---
    plt.figure(figsize=(12, 7))
    max_rounds_loss = 0

    if history_no_attack:
        rounds = len(history_no_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_no_attack['loss'], 'g-s',
                 label=f'No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_under_attack['loss'], 'r-x',
                 label=f'Under Attack (Ran for {rounds} rounds)')

    if history_with_defense:
        rounds = len(history_with_defense['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_with_defense['loss'], 'b-o',
                 label=f'With IDA Defense (Converged in {rounds} rounds)')

    if max_rounds_loss > 0:
        plt.title("Performance Comparison of IDA Online Defense (Loss)")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Loss")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.xlim(0, max_rounds_loss + 5)
        plt.show()
    else:
        print("\n[可视化] 未运行任何场景，无法绘制损失图。")







# -*- coding: utf-8 -*-
"""
@File    : ic_decay.py
@Time    : 2025/9/19
@Author  : dizjia
@Description: IDA+余弦相似度+基于声誉分的指数衰减权重
            没有wandb版本
@History :

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
    "SEED": 42,

    # 实验日志文件路径
    "LOG_FILE_PATH": "./ic_ddw_simulation_log_scene3.csv",

    # 联邦学习设置
    "NUM_CLIENTS": 20,  # 增加客户端总数以更好地模拟统计检测
    "FL_ROUNDS": 200,  # 模拟的总轮数
    "CLIENTS_PER_ROUND": 10,  # 每轮随机挑选10个客户端参与

    # 客户端本地训练参数
    "LOCAL_EPOCHS": 3,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.01,

    # 攻击设置
    "ATTACK_TYPE": "label_flipping",  # "label_flipping" 或 "backdoor"
    "MALICIOUS_CLIENTS": 8,  # 20个客户端里有多少个是恶意的
    "POISON_RATIO": 1,  # 恶意客户端的数据污染率

    # 防御设置 (核心)
    "DEFENSE_ENABLED": True,  # 是否启用在线防御
    "DEFENSE_START_ROUND": 5,  # 从第5轮开始执行防御，给模型一点初始收敛时间
    "SUSPICIOUS_CLIENT_WEIGHT": 0.2, # 可疑客户端在聚合中的权重
    "REPUTATION_THRESHOLD": 3,  # 声誉分累计超过3次则被移除
    "REPUTATION_DECAY_FACTOR": 0.8, # γ值 (gamma)
    "OUTLIER_THRESHOLD": 3.0,  # 判断异常的阈值（超过中位数3个MAD）
    "SIMILARITY_THRESHOLD" :0.2,    #相似度阈值
    "PERTURBATION_STRENGTH": 0.1,  # 扰动强度
    "is_max": False,    # True使用 max,否则使用 mean


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
            self.local_data = LabelFlippingDataset(local_data, poison_ratio=config["POISON_RATIO"])
        self.dataloader = DataLoader(self.local_data, batch_size=config["BATCH_SIZE"], shuffle=True)

    def train(self, global_model_state_dict, learning_rate):
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # [修改] 差异化本地训练轮数
        epochs_to_run = config["LOCAL_EPOCHS"]
        if self.is_malicious:
            epochs_to_run = config["LOCAL_EPOCHS"] * 2  # 例如，恶意客户端的训练轮数是诚实的2倍
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
        float: 模型的不稳定性分数（最大余弦不相似度）。
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

    return np.max(scores)



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
        probe_indices_file = "../ida_simulation/clustered_probe_indices_for_cifar10.npy"
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
        try:
            for t in range(self.config["FL_ROUNDS"]):
                print(f"\n--- 第 {t + 1}/{self.config['FL_ROUNDS']} 轮 ---")

                num_to_select = min(self.config["CLIENTS_PER_ROUND"], len(self.active_clients_pool))
                selected_clients = random.sample(self.active_clients_pool, num_to_select)
                client_updates = {}
                global_model_state_dict = self.global_model.state_dict()
                current_lr = self.optimizer.param_groups[0]['lr']

                for client in selected_clients:
                    update = client.train(copy.deepcopy(global_model_state_dict), learning_rate=current_lr)
                    client_updates[client.client_id] = update

                if self.config["DEFENSE_ENABLED"] and t + 1 >= self.config["DEFENSE_START_ROUND"]:
                    suspicious_ids = []  # 首先初始化一个空的可疑列表
                    # --- 1. 计算不稳定增量 (IDA部分) ---
                    score_base = calculate_instability_score(
                        global_model_state_dict, self.probe_images, self.device, self.config
                    )
                    deltas = {}
                    for client_id, update in client_updates.items():
                        temp_model_state_dict = {
                            key: global_model_state_dict[key] + update[key] for key in global_model_state_dict
                        }
                        score_temp = calculate_instability_score(
                            temp_model_state_dict, self.probe_images, self.device, self.config
                        )
                        deltas[client_id] = score_temp - score_base

                    # --- 2. 计算更新方向相似度 (交叉验证部分) ---
                    similarities = {}
                    if len(client_updates) > 1:
                        all_updates_flat = torch.stack([
                            torch.cat([p.flatten() for p in update.values()])
                            for update in client_updates.values()
                        ])
                        # 使用坐标平均数来定义主流方向
                        avg_update_flat = torch.mean(all_updates_flat, dim=0)

                        for client_id, update in client_updates.items():
                            update_flat = torch.cat([p.flatten() for p in update.values()])
                            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                            similarity = cos(update_flat, avg_update_flat).item()
                            similarities[client_id] = similarity

                    # --- 3.混合判断逻辑，识别本轮的“可疑分子” ---
                    delta_values = list(deltas.values())
                    if len(delta_values) > 1:
                        median = np.median(delta_values)
                        mad = np.median([np.abs(v - median) for v in delta_values])
                        if mad == 0: mad = 1e-9

                        SIMILARITY_THRESHOLD = self.config.get("SIMILARITY_THRESHOLD", 0.0)

                        for client_id, delta in deltas.items():
                            score_mad = (delta - median) / mad
                            similarity_score = similarities.get(client_id, 1.0)
                            if score_mad > self.config["OUTLIER_THRESHOLD"] or similarity_score < SIMILARITY_THRESHOLD:
                                suspicious_ids.append(client_id)

                    if suspicious_ids:
                        print(f"    [侦测模块] 本轮检测到可疑客户端: {suspicious_ids}")

                    # --- 4. [核心修改] 更新永久声誉分 (只增不减) ---
                    for client_id in suspicious_ids:
                        self.reputation_scores[client_id] += 1

                    # print(f"    [声誉模块] 当前所有客户端声誉分: {self.reputation_scores}")

                    # --- 5. [核心修改] 模型加权聚合 (基于“指数衰减”) ---
                    decay_factor = self.config["REPUTATION_DECAY_FACTOR"]
                    weights = {cid: decay_factor ** self.reputation_scores.get(cid, 0)
                               for cid in client_updates.keys()}

                    # print(f"    本轮聚合权重: {weights}")

                    # (加权平均的计算逻辑保持不变)
                    sum_of_weights = sum(weights.values())
                    avg_update = {}
                    for key in client_updates[list(client_updates.keys())[0]].keys():
                        weighted_sum_layer = torch.stack(
                            [client_updates[cid][key] * weights[cid] for cid in client_updates.keys()],
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
    print(f"[实验配置] IDA阈值 = {config["OUTLIER_THRESHOLD"]},相似度阈值 = {config["SIMILARITY_THRESHOLD"]},声誉容忍度 = {config["REPUTATION_THRESHOLD"]},扰动强度 = {config["PERTURBATION_STRENGTH"]}")

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
        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_1 = common_params_to_log.copy()
        # 场景一没有攻击和防御，用 N/A 覆盖
        params_log_1.update({
            "MALICIOUS_CLIENTS":"N/A",
            "POISON_RATIO": "N/A",
            "PERTURBATION_STRENGTH": "N/A",
            "OUTLIER_THRESHOLD": "N/A",
            "REPUTATION_THRESHOLD": "N/A"
        })
        actual_rounds_1 = len(history_no_attack['accuracy'])
        metrics_log_1 = {
            "actual_rounds": actual_rounds_1,
            "final_accuracy": f"{history_no_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A"
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
        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_2 = common_params_to_log.copy()
        params_log_2.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": "N/A",
            "OUTLIER_THRESHOLD": "N/A",
            "REPUTATION_THRESHOLD": "N/A"
        })
        actual_rounds_2 = len(history_under_attack['accuracy'])
        metrics_log_2 = {
            "actual_rounds": actual_rounds_2,
            "final_accuracy": f"{history_under_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A"
        }
        log_experiment_results(config["LOG_FILE_PATH"], "2", params_log_2, metrics_log_2)


    if 3 in scenarios_to_run:
        # --- 场景三: 有攻击, 有您的IDA防御 ---
        print("\n\n=============== 场景三: 有攻击, 启用IDA防御 ===============")



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
        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_3 = common_params_to_log.copy()
        params_log_3.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": config["PERTURBATION_STRENGTH"],
            "OUTLIER_THRESHOLD": config["OUTLIER_THRESHOLD"],
            "REPUTATION_THRESHOLD": config["REPUTATION_THRESHOLD"]
        })
        actual_rounds_3 = len(history_with_defense['accuracy'])
        metrics_log_3 = {
            "actual_rounds": actual_rounds_3,
            "final_accuracy": f"{history_with_defense['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": len(set(history_with_defense['removed_malicious'])),
            "benign_removed": len(set(history_with_defense['removed_benign']))
        }
        log_experiment_results(config["LOG_FILE_PATH"], "3", params_log_3, metrics_log_3)

    # print(f"\n[日志] 所有场景已成功记录到: {config['LOG_FILE_PATH']}")


    # --- 4. 结果可视化 ---
    # [修改] 4. 让绘图代码更健壮，只绘制实际运行过的场景
    plt.figure(figsize=(10, 6))
    max_rounds = 0

    if history_no_attack:
        rounds = len(history_no_attack['accuracy'])
        max_rounds = max(max_rounds, rounds)
        plt.plot(range(1, rounds + 1), history_no_attack['accuracy'], 'g-s',
                 label=f'No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['accuracy'])
        max_rounds = max(max_rounds, rounds)
        plt.plot(range(1, rounds + 1), history_under_attack['accuracy'], 'r-x',
                 label=f'Under Attack (Ran for {rounds} rounds)')

    # --- [核心修改] 防御效果总结 (适配新的“永久降权”逻辑) ---
    if history_with_defense:
        print("\n--- 防御效果总结 ---")

        # 从server对象中获取最终被永久降权的客户端ID集合
        convicted_clients = server_with_defense.permanently_downweighted_clients

        # 判断其中哪些是恶意的，哪些是良性的
        malicious_clients_config = config_with_defense["MALICIOUS_CLIENTS"]
        downweighted_malicious = sorted([cid for cid in convicted_clients if cid < malicious_clients_config])
        downweighted_benign = sorted([cid for cid in convicted_clients if cid >= malicious_clients_config])

        print(f"被成功永久降权的恶意客户端 ({len(downweighted_malicious)}个): {downweighted_malicious}")
        print(f"被错误永久降权的良性客户端 ({len(downweighted_benign)}个): {downweighted_benign}")

    # 只有在未运行任何场景时，才打印提示
    if not history_no_attack and not history_under_attack and not history_with_defense:
        print("\n[可视化] 未运行任何场景，无需绘图。")

    # 只有在至少绘制了一条线时才显示图表
    if max_rounds > 0:
        plt.title("Performance Comparison of IDA Online Defense")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlim(0, max_rounds + 5)
        plt.show()

    # --- 图表二：损失值对比 ---
    plt.figure(figsize=(12, 7))
    max_rounds_loss = 0

    if history_no_attack:
        rounds = len(history_no_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        # 注意：这里的数据源改为了 history_no_attack['loss']
        plt.plot(range(1, rounds + 1), history_no_attack['loss'], 'g-s',
                 label=f'No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        # 注意：这里的数据源改为了 history_under_attack['loss']
        plt.plot(range(1, rounds + 1), history_under_attack['loss'], 'r-x',
                 label=f'Under Attack (Ran for {rounds} rounds)')

    if history_with_defense:
        rounds = len(history_with_defense['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        # 注意：这里的数据源改为了 history_with_defense['loss']
        plt.plot(range(1, rounds + 1), history_with_defense['loss'], 'b-o',
                 label=f'With IDA Defense (Converged in {rounds} rounds)')

    # 只有在至少绘制了一条线时才显示图表
    if max_rounds_loss > 0:
        # 修改图表的标题和Y轴标签
        plt.title("Performance Comparison of IDA Online Defense (Loss)")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Loss")
        plt.legend()
        plt.grid(True)
        # 设置Y轴下限为0，让matplotlib自动适应上限
        plt.ylim(bottom=0)
        plt.xlim(0, max_rounds_loss + 5)
        plt.show()






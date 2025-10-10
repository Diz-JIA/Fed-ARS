import torch
import torch.optim as optim
import numpy as np
import random
import copy
import os

from .models import SimpleCNN
from .utils import evaluate_model
from .attack_strategies import attack_strategy_factory
from .defense import calculate_adversarial_vulnerability, clip_update_norm_


class Server:
    def __init__(self, all_clients, test_loader, config):
        self.device = config["DEVICE"]
        self.config = config
        self.global_model = SimpleCNN().to(self.device)
        self.test_loader = test_loader
        self.all_clients = all_clients
        self.active_clients_pool = list(all_clients)
        self.reputation_scores = {client.client_id: 0 for client in all_clients}
        self.malicious_ids = {c.client_id for c in all_clients if c.is_malicious}
        self.probe_images = self._generate_probes()
        self.history = {'accuracy': [], 'loss': []}

        # 初始化优化器、调度器和早停变量
        self.optimizer = optim.SGD(self.global_model.parameters(), lr=self.config["LEARNING_RATE"])
        self.scheduler = self._initialize_scheduler()
        self.patience_counter = 0
        self.best_performance_metric = float('inf') if self.config.get("EARLY_STOPPING_METRIC") == 'loss' else 0.0
        self.best_model_state_dict = None

        # 初始化攻击策略
        attack_type = self.config.get("ATTACK_TYPE", "backdoor")
        if attack_type in attack_strategy_factory:
            self.attack_strategy = attack_strategy_factory[attack_type]()
        else:
            raise ValueError(f"未知的攻击类型: {attack_type}")

    # --- 主流程函数 ---
    def run_simulation(self):
        """运行完整的联邦学习模拟过程 (重构后)"""
        try:
            for t in range(self.config["FL_ROUNDS"]):
                print(f"\n--- 第 {t + 1}/{self.config['FL_ROUNDS']} 轮 ---")

                # 1. 选择客户端
                selected_clients = self._select_clients()

                # 2. 客户端本地训练并收集更新
                client_updates = self._dispatch_and_collect_updates(selected_clients)

                # 3. (如果需要) 服务器端模拟攻击
                client_updates = self._simulate_server_side_attack(client_updates)

                suspicious_ids = []
                # 4. 执行防御 (如果启用)
                if self.config["DEFENSE_ENABLED"] and t + 1 >= self.config["DEFENSE_START_ROUND"]:
                    suspicious_ids = self._run_defense(client_updates)
                    self._update_reputations_and_remove_clients(suspicious_ids)

                # 5. 聚合模型更新
                avg_update = self._aggregate_updates(client_updates, suspicious_ids)

                # 6. 更新全局模型
                self._update_global_model(avg_update)

                # 7. 评估、记录日志、检查早停
                test_loss, _ = self._evaluate_and_log(t)

                if self._check_early_stopping(test_loss):
                    print(f"    !! 早停触发: 训练在第 {t + 1} 轮终止 !!")
                    break

        except KeyboardInterrupt:
            print("\n\n[用户中断] 已捕获 Ctrl+C 信号...")
            if self.best_model_state_dict is None:
                self.best_model_state_dict = copy.deepcopy(self.global_model.state_dict())

        # 8. 训练结束后的收尾工作
        self._post_training_actions()

        return self.history

    # --- 私有辅助函数 ---
    def _run_defense(self, client_updates):

        # --- “双重审查”防御模型 ---
        # --- Part 0: 预处理 (梯度裁剪) ---
        CLIP_MAX_NORM = self.config["CLIP_MAX_NORM"]  # 这是一个超参数，您可以按需调整
        # 注意：我们克隆一份用于裁剪，以防未来需要原始更新
        clipped_updates = copy.deepcopy(client_updates)
        for client_id in clipped_updates.keys():
            clip_update_norm_(clipped_updates[client_id], CLIP_MAX_NORM)
        # print(f"    [防御流程] 已对更新执行范数裁剪 (上限={CLIP_MAX_NORM})，用于后续分析。")
        client_ids = list(clipped_updates.keys())
        num_clients_in_round = len(client_ids)

        # --- Part 1: 行为异常检测 (基于两两欧氏距离) ---
        behavioral_suspects = []
        if num_clients_in_round > 1:
            updates_flat = {cid: torch.cat([p.flatten() for p in upd.values()]) for cid, upd in
                            clipped_updates.items()}
            distance_matrix = np.zeros((num_clients_in_round, num_clients_in_round))
            for i in range(num_clients_in_round):
                for j in range(i + 1, num_clients_in_round):
                    cid_i, cid_j = client_ids[i], client_ids[j]
                    distance = torch.norm(updates_flat[cid_i] - updates_flat[cid_j], p=2).item()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            isolation_scores = np.sum(distance_matrix, axis=1)

            median_iso = np.median(isolation_scores)
            mad_iso = np.median(np.abs(isolation_scores - median_iso))
            if mad_iso == 0: mad_iso = 1e-9
            score_threshold = self.config["DISTANCE_MAD_THRESHOLD"]
            for i in range(num_clients_in_round):
                if (isolation_scores[i] - median_iso) / mad_iso > score_threshold:
                    behavioral_suspects.append(client_ids[i])

        print(f"    [调试-行为/距离] 检测到嫌疑: {sorted(behavioral_suspects)}")

        # --- Part 2: 后果异常检测 (基于“相对”对抗性脆弱度) ---
        # 2.1 首先，计算当前全局模型的基准脆弱度
        global_model_state_dict = self.global_model.state_dict()
        vulnerability_base = calculate_adversarial_vulnerability(
            global_model_state_dict, self.probe_images, self.device, self.config
        )

        # 2.2 计算每个更新引入的“脆弱度增量”
        vulnerability_deltas = {}
        for cid, update in client_updates.items():
            temp_model_state = {k: global_model_state_dict[k] + update[k] for k in update}
            vulnerability_post = calculate_adversarial_vulnerability(
                temp_model_state, self.probe_images, self.device, self.config
            )
            vulnerability_deltas[cid] = vulnerability_post - vulnerability_base

        # [调试功能] 打印本轮所有客户端的“相对脆弱度”分数排名
        delta_list = sorted(list(vulnerability_deltas.items()), key=lambda item: item[1], reverse=True)
        print("    [调试-相对脆弱度排名] (Client ID, Score, Type):")
        for client_id, score in delta_list:
            client_type = "恶意" if client_id in self.malicious_ids else "良性"
            print(f"      - Client {client_id:<2} | Score: {score:8.4f} | Type: {client_type}")
        # 2.3 使用MAD检测“增量”分数中的“异常低值”
        consequential_suspects = []
        delta_values = list(vulnerability_deltas.values())
        if len(delta_values) > 1:
            median_delta = np.median(delta_values)
            mad_delta = np.median(np.abs(delta_values - median_delta))
            if mad_delta == 0: mad_delta = 1e-9

            # 我们依然使用VULNERABILITY_MAD_THRESHOLD，但现在它作用于“增量”
            score_threshold = self.config["VULNERABILITY_MAD_THRESHOLD"]
            for cid, delta in vulnerability_deltas.items():
                z_score = (delta - median_delta) / mad_delta
                # 预期信号依然是反转的，所以我们寻找的是远低于中位数的客户端
                if z_score < -score_threshold:
                    consequential_suspects.append(cid)

        print(f"    [调试-后果/相对脆弱度] 检测到嫌疑: {sorted(consequential_suspects)}")

        # --- Part 3: 取并集，最终裁决 (OR Logic) ---
        # suspicious_ids = sorted(list(set(behavioral_suspects) & set(consequential_suspects)))
        suspicious_ids = sorted(list(set(behavioral_suspects) | set(consequential_suspects)))

        if suspicious_ids:
            print(f"    [侦测模块] 本轮最终可疑客户端: {suspicious_ids}")

        return suspicious_ids

    def _update_reputations_and_remove_clients(self, suspicious_ids):
        for client_id in suspicious_ids:
            if client_id in self.reputation_scores:
                self.reputation_scores[client_id] += 1

        # 检查并移除客户端
        clients_to_remove = [cid for cid, score in self.reputation_scores.items()
                             if score > self.config["REPUTATION_THRESHOLD"]]

        if clients_to_remove:
            initial_pool_size = len(self.active_clients_pool)
            self.active_clients_pool = [c for c in self.active_clients_pool if c.client_id not in clients_to_remove]

            # 从声誉分中也移除，防止重复判断
            for cid in clients_to_remove:
                del self.reputation_scores[cid]

            print(f"    [防御系统] 客户端 {clients_to_remove} 声誉分超过阈值，已被永久移除。")
            print(f"    [防御系统] 剩余活跃客户端数量: {len(self.active_clients_pool)}")

    def _aggregate_updates(self, client_updates, suspicious_ids):
        # 裁剪更新用于聚合
        clipped_updates = copy.deepcopy(client_updates)
        for cid in clipped_updates:
            clip_update_norm_(clipped_updates[cid], 1.2)  # 这里的1.2可以放入config

        # 聚合逻辑
        decay_factor = self.config["REPUTATION_DECAY_FACTOR"]
        weights = {cid: decay_factor ** self.reputation_scores.get(cid, 0) for cid in clipped_updates.keys()}

        sum_of_weights = sum(weights.values())
        if sum_of_weights == 0: return None

        avg_update = {}
        for key in clipped_updates[list(clipped_updates.keys())[0]].keys():
            weighted_sum_layer = torch.stack(
                [clipped_updates[cid][key] * weights[cid] for cid in clipped_updates.keys()], dim=0
            ).sum(dim=0)
            avg_update[key] = weighted_sum_layer / sum_of_weights
        return avg_update

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

    def _update_global_model(self, avg_update):
        if avg_update:
            with torch.no_grad():
                for param_name, param in self.global_model.named_parameters():
                    param.data += avg_update[param_name]

    def _evaluate_and_log(self, round_num):
        test_loss, accuracy = evaluate_model(self.global_model, self.test_loader, self.device)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(test_loss)

        current_lr = self.optimizer.param_groups[0]['lr']
        print(
            f"    >> 第 {round_num} 轮结束 | 当前LR: {current_lr:.6f} | 测试损失: {test_loss:.4f} | 全局模型准确率: {accuracy * 100:.2f}%")

        if self.scheduler:
            self.scheduler.step(test_loss)

        return test_loss, accuracy

    def _check_early_stopping(self, current_loss):
        if not self.config.get("EARLY_STOPPING_ENABLED", False):
            return False

        # 这里的逻辑假设了监控loss，您可以根据需要扩展
        if current_loss < self.best_performance_metric:
            self.best_performance_metric = current_loss
            self.patience_counter = 0
            self.best_model_state_dict = copy.deepcopy(self.global_model.state_dict())
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config["EARLY_STOPPING_PATIENCE"]

    def _post_training_actions(self):
        if self.best_model_state_dict:
            self.global_model.load_state_dict(self.best_model_state_dict)
            print("\n已加载性能最佳的模型状态。")

        save_path = f"./saved_models/{self.config.get('SCENARIO_NAME', 'model')}_final.pth"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.global_model.state_dict(), save_path)
        print(f"当前场景的最终模型已保存至: {save_path}")

    def _initialize_scheduler(self):
        if not self.config.get("SCHEDULER_ENABLED", False):
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config["SCHEDULER_FACTOR"],
            patience=self.config["SCHEDULER_PATIENCE"],
        )

    def _select_clients(self):
        num_to_select = min(self.config["CLIENTS_PER_ROUND"], len(self.active_clients_pool))
        return random.sample(self.active_clients_pool, num_to_select)

    def _dispatch_and_collect_updates(self, selected_clients):
        client_updates = {}
        global_model_state_dict = copy.deepcopy(self.global_model.state_dict())
        current_lr = self.optimizer.param_groups[0]['lr']

        for client in selected_clients:
            update = client.train(global_model_state_dict, current_lr, self.config)
            client_updates[client.client_id] = update
        return client_updates

    def _simulate_server_side_attack(self, client_updates):
        honest_updates = {cid: upd for cid, upd in client_updates.items() if cid not in self.malicious_ids}
        return self.attack_strategy.execute(client_updates, self.malicious_ids, honest_updates)



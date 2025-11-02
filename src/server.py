import torch
import torch.optim as optim
import numpy as np
import random
import copy
import os
import time

from . import models
from .utils import evaluate_model
from .attack_strategies import attack_strategy_factory
from .defense import calculate_adversarial_vulnerability, clip_update_norm_, calculate_sparsity_score
from . import aggregation_rules


class Server:
    def __init__(self, all_clients, test_loader, config):
        self.device = config["DEVICE"]
        self.config = config
        self.test_loader = test_loader
        self.all_clients = all_clients
        self.active_clients_pool = list(all_clients)
        self. reputation_scores = {client.client_id: 0 for client in all_clients}
        self.malicious_ids = {c.client_id for c in all_clients if c.is_malicious}
        self.probe_images = self._generate_probes()
        self.history = {'accuracy': [], 'loss': []}
        self.total_malicious_removed = 0
        self.total_benign_removed = 0
        # 记录所有恶意客户端被移除的轮次，-1 代表尚未发生
        self.all_malicious_removed_round = -1

        model_name = self.config.get("MODEL_NAME", "simple_cnn")  # 默认为 simple_cnn
        aggregation_method = self.config.get("AGGREGATION_METHOD", "avg")
        # --- [核心修改] 动态实例化模型 ---
        if model_name in models.model_factory:
            model_constructor = models.model_factory[model_name]

            # 从 config 中获取数据集特定参数
            in_channels = self.config["in_channels"]
            num_classes = self.config["num_classes"]

            # 将参数传递给构造函数
            self.global_model = model_constructor(
                in_channels=in_channels,
                num_classes=num_classes
            ).to(self.device)

            print(
                f"    [服务器设置] 模型: {model_name.upper()} , 聚合: {aggregation_method.upper()}")
        else:
            raise ValueError(f"未知的模型名称: {model_name}")

        # 初始化优化器、调度器和早停变量
        weight_decay = self.config.get("WEIGHT_DECAY", 5e-4)
        self.optimizer = optim.SGD(self.global_model.parameters(), lr=self.config["LEARNING_RATE"],
                                   weight_decay=weight_decay, momentum=0.9)
        self.scheduler = self._initialize_scheduler()
        self.patience_counter = 0
        self.best_performance_metric = float('inf') if self.config.get("EARLY_STOPPING_METRIC") == 'loss' else 0.0
        self.best_model_state_dict = None

        # 初始化攻击策略
        attack_type = self.config.get("ATTACK_TYPE", "backdoor")
        if attack_type in attack_strategy_factory:
            # 将 self.config 传递给攻击策略的构造函数
            self.attack_strategy = attack_strategy_factory[attack_type](self.config)
        else:
            raise ValueError(f"未知的攻击类型: {attack_type}")

    # --- 主流程函数 ---
    def run_simulation(self):
        """运行完整的联邦学习模拟过程 (重构后)"""
        total_duration = 0.0
        try:
            for t in range(self.config["FL_ROUNDS"]):
                print(f"\n--- 第 {t + 1}/{self.config['FL_ROUNDS']} 轮 ---")
                start_time = time.time()  # <--- [新增] 记录本轮开始时间

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
                    self._update_reputations_and_remove_clients(suspicious_ids,t)

                # 5. 聚合模型更新 (根据配置选择不同策略)
                aggregation_method = self.config.get("AGGREGATION_METHOD", "avg")  # 默认为您原来的方法
                avg_update = None

                # 如果防御开启 (场景三)，强制使用你带裁剪和声誉的聚合器
                if self.config["DEFENSE_ENABLED"]:
                    print("    [服务器聚合] 防御开启，使用“裁剪+声誉加权平均”。")
                    # _aggregate_updates 包含裁剪，是防御的一部分
                    avg_update = self._aggregate_updates(client_updates, suspicious_ids)

                # 如果防御关闭 (场景一、二)
                else:
                    # 如果聚合方法是 'avg'
                    if aggregation_method == "avg":
                        print("    [服务器聚合] 使用FedAvg。")
                        # 调用我们新增的、真正无防御的聚合器
                        avg_update = self._naive_average_updates(client_updates)

                    # 如果是 'trimmed_mean' 等其他鲁棒聚合
                    elif aggregation_method in aggregation_rules.aggregation_factory:
                        print(f"    [服务器聚合] 使用鲁棒聚合: {aggregation_method}。")
                        agg_func = aggregation_rules.aggregation_factory[aggregation_method]
                        avg_update = agg_func(client_updates, self.config)
                    else:
                        raise ValueError(f"未知的聚合方法: {aggregation_method}")

                # --- [新增最终调试代码] ---
                # 在更新全局模型前，检查聚合更新 avg_update 的状态
                if avg_update:
                    # 1. 获取模型中所有可训练参数的名称
                    trainable_param_names = {name for name, _ in self.global_model.named_parameters()}

                    # 2. 从聚合更新中，只筛选出属于可训练参数的“更新增量 (Deltas)”
                    param_deltas = [p for key, p in avg_update.items() if key in trainable_param_names]

                    # 3. 只对这些增量计算范数，这才是真正有意义的监控指标
                    if param_deltas:
                        flat_deltas = torch.cat([p.flatten() for p in param_deltas])
                        delta_norm = torch.norm(flat_deltas, p=2)
                        # 你会发现，这个值应该和你设置的裁剪阈值 (CLIP_MAX_NORM: 1.2) 非常接近
                        print(f"    [服务器调试] 参数更新范数 (Delta Norm): {delta_norm.item():.4f}")
                    else:
                        print("    [服务器调试] 聚合更新中没有参数增量。")
                # --- [调试代码结束] ---

                # 6. 更新全局模型
                self._update_global_model(avg_update)

                # 为了让学习率调度器(scheduler)能够正常工作并消除PyTorch警告，
                # 我们在这里空跑一次优化器步骤。
                self.optimizer.zero_grad()  # 确保没有残留梯度
                self.optimizer.step()  # 执行一个空步骤

                # 7. 评估、记录日志、检查早停
                test_loss, _ = self._evaluate_and_log(t)

                end_time = time.time()  # <--- [新增] 记录本轮结束时间
                duration = end_time - start_time  # <--- [新增] 计算本轮耗时
                total_duration += duration  # <--- [新增] 累加到总耗时
                print(f"    >> 本轮耗时: {duration:.2f} 秒")  # <--- [新增] 打印本轮耗时

                if self._check_early_stopping(test_loss):
                    print(f"    !! 早停触发: 训练在第 {t + 1} 轮终止 !!")
                    break

        except KeyboardInterrupt:
            print("\n\n[用户中断] 已捕获 Ctrl+C 信号...")
            if self.best_model_state_dict is None:
                self.best_model_state_dict = copy.deepcopy(self.global_model.state_dict())

        # 8. 训练结束后的收尾工作
        self._post_training_actions()

        # --- [新增] 打印总耗时信息 ---
        print("\n--- 训练结束 ---")
        total_minutes = total_duration / 60
        avg_round_time = total_duration / self.config['FL_ROUNDS'] if self.config['FL_ROUNDS'] > 0 else 0
        print(f"总耗时: {total_duration:.2f} 秒 ({total_minutes:.2f} 分钟)")
        print(f"平均每轮耗时: {avg_round_time:.2f} 秒")
        # --- [新增结束] ---

        return self.history

    # --- 私有辅助函数 ---
    def _run_defense(self, client_updates):
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
        # 2.3 使用MAD检测“增量”分数中的“异常高/低值”
        consequential_suspects = []
        delta_values = list(vulnerability_deltas.values())

        # --- [新增] 用于详细调试打印的列表 ---
        client_debug_info = []
        if len(delta_values) > 1:
            median_delta = np.median(delta_values)
            mad_delta = np.median(np.abs(delta_values - median_delta))
            if mad_delta == 0: mad_delta = 1e-9
            print(f"      - Median_Delta: {median_delta} | Mad_Delta: {mad_delta}")

            base_threshold = self.config["VULNERABILITY_MAD_THRESHOLD"]
            num_total_clients = self.config["NUM_CLIENTS"]
            num_active_clients = len(self.active_clients_pool)

            # 客户端越少，阈值越宽松 (这里使用线性增长，也可以用其他函数)
            # 当客户端数量从总数降低到0时，阈值从 base_threshold 增长到 base_threshold * （K+1）
            k_factor = self.config.get("DYNAMIC_THRESHOLD_K", 1.0)
            scaling_factor = 1.0 + k_factor * (1.0 - (num_active_clients / num_total_clients))  # <-- [修改后]
            score_threshold = base_threshold * scaling_factor
            # 可以在这里打印一下动态阈值，方便调试
            print(f"    [调试-动态阈值] 活跃客户端: {num_active_clients}, k因子: {k_factor:.1f}, 缩放: {scaling_factor:.2f}, 最终阈值: {score_threshold:.2f}")

            for cid, delta in vulnerability_deltas.items():
                z_score = (delta - median_delta) / mad_delta
                is_suspicious = abs(z_score) > score_threshold
                client_type = "恶意" if cid in self.malicious_ids else "良性"

                client_debug_info.append({
                    "id": cid,
                    "score": delta,
                    "z_score": z_score,
                    "type": client_type,
                    "suspicious": is_suspicious
                })

                if is_suspicious:
                    consequential_suspects.append(cid)

            """
            # 按 Z-Score 的绝对值降序排列，方便查看异常值
            client_debug_info.sort(key=lambda x: abs(x["z_score"]), reverse=True)
            print("    [调试-Z-Score排名] (Client ID, Score, Z-Score, Type, Suspicious):")
            for info in client_debug_info:
                # 如果被标记为嫌疑，则在行尾添加标记
                susp_marker = "<- [嫌疑]" if info["suspicious"] else ""
                print(
                    f"      - Client {info['id']:<2} | Score: {info['score']:8.4f} | Z-Score: {info['z_score']: 8.4f} | Type: {info['type']:<2} | {susp_marker}")
            """
        else:  # delta_values <= 1
            print("    [调试-后果/相对脆弱度] 客户端数量不足 (<=1)，无法计算 MAD 和 Z-Score。")

        print(f"    [调试-后果/相对脆弱度] 检测到嫌疑: {sorted(consequential_suspects)}")

        suspicious_ids = sorted(list(set(consequential_suspects)))

        if suspicious_ids:
            print(f"    [侦测模块] 本轮最终可疑客户端: {suspicious_ids}")

        return suspicious_ids

    def _update_reputations_and_remove_clients(self, suspicious_ids, current_round):
        for client_id in suspicious_ids:
            if client_id in self.reputation_scores:
                self.reputation_scores[client_id] += 1

        # --- [新增逻辑] 为表现良好的客户端降低声誉分 ---
        # 1. 获取本轮所有活跃且未被怀疑的客户端ID
        active_ids_in_round = {c.client_id for c in self.active_clients_pool if
                               c.client_id in self.reputation_scores}
        non_suspicious_ids = active_ids_in_round - set(suspicious_ids)

        # 2. 对这些客户端的声誉分进行衰减
        decay_amount = 1  # 这个值可以放入config中，代表“洗白”的速度
        for cid in non_suspicious_ids:
            self.reputation_scores[cid] = max(0, self.reputation_scores[cid] - decay_amount)

        # 检查并移除客户端
        clients_to_remove = [cid for cid, score in self.reputation_scores.items()
                             if score > self.config["REPUTATION_THRESHOLD"]]

        if clients_to_remove:
            # --- [核心修改] 将待移除客户端分类 ---
            removed_malicious = []
            removed_benign = []
            for cid in clients_to_remove:
                if cid in self.malicious_ids:
                    removed_malicious.append(cid)
                else:
                    removed_benign.append(cid)

            # --- 从活跃池和声誉字典中执行移除操作 ---
            self.active_clients_pool = [c for c in self.active_clients_pool if c.client_id not in clients_to_remove]
            for cid in clients_to_remove:
                del self.reputation_scores[cid]

            # --- [核心修改] 分类打印移除信息 ---
            if removed_malicious:
                print(f"    [防御系统] 成功移除恶意客户端: {sorted(removed_malicious)}")
            if removed_benign:
                # 使用醒目的 "警告" 标签提示误伤
                print(f"    [防御系统-警告] 误伤良性客户端: {sorted(removed_benign)}")

            self.total_malicious_removed += len(removed_malicious)
            self.total_benign_removed += len(removed_benign)

            # --- [新增] 检查是否所有恶意客户端都已被移除 ---
            # 1. 检查是否是第一次达到 "全部移除" (self.all_malicious_removed_round == -1)
            # 2. 检查总移除数是否已达到初始恶意总数
            if self.all_malicious_removed_round == -1 and \
                    self.total_malicious_removed == self.config.get("MALICIOUS_CLIENTS", 0):
                # 记录轮次 (t 是 0-indexed, 所以 +1)
                self.all_malicious_removed_round = current_round + 1
                print(f"    [防御系统-里程碑] "
                      f"所有 {self.total_malicious_removed} 个恶意客户端已在第 {self.all_malicious_removed_round} 轮被全部移除！")

            print(f"    [防御系统] 剩余活跃客户端数量: {len(self.active_clients_pool)}")

    def _aggregate_updates(self, client_updates, suspicious_ids):
        # --- [核心修复：智能裁剪] ---
        clipped_updates = copy.deepcopy(client_updates)
        trainable_param_names = {name for name, _ in self.global_model.named_parameters()}
        clip_norm_threshold = self.config.get("CLIP_MAX_NORM", 1.2)  # 从配置中读取裁剪阈值

        for cid in clipped_updates:
            # 1. 从更新字典中分离出可训练参数的 delta
            param_deltas = {key: val for key, val in clipped_updates[cid].items() if key in trainable_param_names}

            if not param_deltas:
                continue  # 如果没有可训练参数，则跳过

            # 2. 只基于这些 delta 计算范数
            flat_deltas = torch.cat([p.flatten() for p in param_deltas.values()])
            total_norm = torch.norm(flat_deltas, p=2)

            # 3. 如果需要，计算裁剪因子并只应用在 delta 上
            if total_norm > clip_norm_threshold:
                clip_factor = clip_norm_threshold / (total_norm + 1e-6)
                for key in param_deltas.keys():
                    clipped_updates[cid][key].mul_(clip_factor)  # 原地应用裁剪

        # --- [修复结束] ---
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

    # --- [新增] 真正“无防御”的聚合器 ---
    def _naive_average_updates(self, client_updates):
        """
        一个纯粹的 FedAvg 聚合器，【没有】任何范数裁剪或加权。
        """
        if not client_updates:
            return None

        avg_update = {}
        # 获取一个模板 (来自任意一个客户端)
        sample_update = client_updates[list(client_updates.keys())[0]]

        for key in sample_update.keys():
            # 堆叠所有客户端的同一层更新，然后在第0维 (客户端维度) 求均值
            avg_update[key] = torch.stack(
                [client_updates[cid][key] for cid in client_updates.keys()],
                dim=0
            ).mean(dim=0)

        return avg_update

    def _generate_probes(self):
        # --- [核心修改] 动态加载探针文件 ---
        # 从 config 中获取正确的探针文件名
        probe_indices_file = self.config["probe_file"]

        try:
            indices = np.load(probe_indices_file)
            print(f"    [服务器设置] 成功从 {probe_indices_file} 加载了 {len(indices)} 个探针。")
        except FileNotFoundError:
            print(f"    [错误] 探针索引文件 {probe_indices_file} 未找到！")
            print(f"    [提示] 请确认您已经运行了 'create_probes_for_{self.config['DATASET_NAME']}.py' 脚本。")
            raise
        # --- [修改结束] ---

        probe_images = []
        for idx in indices:
            img, _ = self.test_loader.dataset[int(idx)]
            probe_images.append(img)
        return probe_images

    def _update_global_model(self, avg_update):
        """
    使用直接相加的方式更新模型参数，并手动加载缓冲区状态。
    这是最稳健的联邦平均实现方式。
    """
        if avg_update:
            with torch.no_grad():
                # --- 第1部分：直接、手动地更新可训练参数 ---
                for name, param in self.global_model.named_parameters():
                    if name in avg_update:
                        # 使用 in-place 的 add_ 操作，效率高
                        param.data.add_(avg_update[name])

                # --- 第2部分：手动更新状态缓冲区 (逻辑保持不变) ---
                trainable_param_names = {name for name, _ in self.global_model.named_parameters()}
                current_state_dict = self.global_model.state_dict()
                for key in avg_update.keys():
                    if key not in trainable_param_names:
                        current_state_dict[key] = avg_update[key]

                # 重新加载以确保缓冲区状态被正确更新
                # 尽管参数是in-place更新的，但为了统一和安全，这里重新加载整个state_dict
                self.global_model.load_state_dict(current_state_dict)

    def _evaluate_and_log(self, round_num):
        test_loss, accuracy = evaluate_model(self.global_model, self.test_loader, self.device)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(test_loss)

        current_lr = self.optimizer.param_groups[0]['lr']
        print(
            f"    >> 第 {round_num + 1} 轮结束 | 当前LR: {current_lr:.6f} | 测试损失: {test_loss:.4f} | 全局模型准确率: {accuracy * 100:.2f}%")

        if self.scheduler:
            # MultiStepLR的step不需要参数，而ReduceLROnPlateau需要
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()

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
        scheduler_type = self.config.get("SCHEDULER_TYPE", "plateau")  # 默认为旧方式

        if scheduler_type == "multistep":
            print("    [服务器设置] 使用 MultiStepLR 学习率调度器。")
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config["SCHEDULER_MILESTONES"],
                gamma=self.config["SCHEDULER_GAMMA"]
            )
        else:  # 默认或未指定时，使用原来的 ReduceLROnPlateau
            print("    [服务器设置] 使用 ReduceLROnPlateau 学习率调度器。")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config["SCHEDULER_FACTOR"],
                patience=self.config["SCHEDULER_PATIENCE"],
            )

    def _select_clients(self):
        """
                选择客户端，并确保每轮选中的恶意客户端数量【严格小于】一半。
                """
        num_to_select = self.config["CLIENTS_PER_ROUND"]

        # --- [核心修改] ---
        # 计算严格小于一半的最大恶意客户端数量
        max_malicious_allowed = (num_to_select - 1) // 2
        # --- 修改结束 ---

        # 1. 将活跃客户端池分为良性 H 和恶意 M 两个子池
        benign_pool = [c for c in self.active_clients_pool if not c.is_malicious]
        malicious_pool = [c for c in self.active_clients_pool if c.is_malicious]

        # 2. 确定要从每个池中抽样的数量
        num_malicious_to_sample = min(max_malicious_allowed, len(malicious_pool))
        num_benign_to_sample = min(num_to_select - num_malicious_to_sample, len(benign_pool))

        # 3. 从两个池中分别进行随机抽样
        selected_malicious = random.sample(malicious_pool, num_malicious_to_sample)
        selected_benign = random.sample(benign_pool, num_benign_to_sample)

        # 4. 合并并打乱顺序
        selected_clients = selected_malicious + selected_benign
        random.shuffle(selected_clients)

        # 攻击监控日志 (逻辑不变)
        if self.config.get("MALICIOUS_CLIENTS", 0) > 0:
            actual_malicious_ids = sorted([c.client_id for c in selected_clients if c.is_malicious])
            print(f"    [攻击监控] 本轮选中了 {len(actual_malicious_ids)} 个恶意客户端: {actual_malicious_ids}")

        return selected_clients

    def _dispatch_and_collect_updates(self, selected_clients):
        client_updates = {}
        global_model_state_dict = copy.deepcopy(self.global_model.state_dict())
        current_lr = self.optimizer.param_groups[0]['lr']

        for client in selected_clients:
            update = client.train(global_model_state_dict, current_lr)
            client_updates[client.client_id] = update
        return client_updates

    def _simulate_server_side_attack(self, client_updates):
        honest_updates = {cid: upd for cid, upd in client_updates.items() if cid not in self.malicious_ids}
        return self.attack_strategy.execute(client_updates, self.malicious_ids, honest_updates)



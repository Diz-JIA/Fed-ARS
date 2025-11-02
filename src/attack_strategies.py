import torch
import copy
from itertools import combinations
import numpy as np


class AttackStrategy:
    """攻击策略的基类 (接口)"""

    def __init__(self, config=None):
        self.config = config if config is not None else {}

    def execute(self, client_updates, malicious_ids, honest_updates):
        """执行攻击策略，返回处理后的 client_updates"""
        raise NotImplementedError


class NoServerSideAttack(AttackStrategy):
    """一个“空”策略，用于客户端攻击（如后门），服务器端无需操作。"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        # print("    [攻击模拟] 客户端侧攻击，服务器端无操作。")
        return client_updates  # 直接返回原始更新


# --- Helper Function: Flatten Update ---
def flatten_update(update_dict):
    """将更新字典展平成一个单一的 Torch 张量向量。"""
    # 确保只展平浮点类型的张量，并保持顺序一致
    return torch.cat([p.flatten() for p in update_dict.values() if p.is_floating_point()])


# --- Helper Function: Unflatten Update ---
def unflatten_update(flat_vector, model_template_dict):
    """将展平的向量恢复为原始字典结构。"""
    unflattened = {}
    current_index = 0
    # 必须严格按照模板的键和形状来恢复
    for key, template_tensor in model_template_dict.items():
        if template_tensor.is_floating_point():
            num_elements = template_tensor.numel()
            # 从向量中切片，并重塑为模板张量的形状
            unflattened[key] = flat_vector[current_index: current_index + num_elements].view_as(template_tensor)
            current_index += num_elements
        else:
            # 对于非浮点类型（例如 BatchNorm 的 num_batches_tracked），直接复制模板值
            unflattened[key] = template_tensor.clone()

            # 检查是否所有元素都被使用了
    if current_index != flat_vector.numel():
        print(f"[警告] Unflatten 过程中似乎有维度不匹配！ {current_index} vs {flat_vector.numel()}")
        # 可以选择抛出错误或继续
        # raise ValueError("维度不匹配")

    return unflattened

class MinMaxAttack(AttackStrategy):
    """
    实现 Min-Max 攻击。
    恶意梯度试图最大化与诚实梯度均值的距离，
    但其到任何诚实梯度的最大距离，不超过诚实梯度集群的直径的 beta 倍。
    beta 是一个可选配置参数 (MINMAX_BETA)，默认为 1.0。
    """
    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]
        honest_client_ids = list(honest_updates.keys())

        # 必须有至少2个诚实客户端才能计算直径，且必须有恶意客户端
        if len(honest_client_ids) < 2 or not malicious_client_ids:
            if not malicious_client_ids:
                 print("    [攻击模拟] MinMax: 没有恶意客户端参与本轮。")
            elif len(honest_client_ids) < 2:
                 print("    [攻击模拟] MinMax: 诚实客户端少于2个，无法计算集群直径，跳过攻击。")
            return client_updates # 无法执行攻击，返回原样

        print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，"
              f"基于 {len(honest_client_ids)} 个诚实客户端，开始制作 MinMax 攻击...")

        # --- 0. 从 config 获取 beta 参数 ---
        beta = float(self.config.get("MINMAX_BETA", 1.0)) # 默认为 1.0
        # print(f"    [攻击模拟] MinMax: 使用 beta = {beta:.2f}")

        # --- 1. 展平所有诚实更新 ---
        model_template_dict = honest_updates[honest_client_ids[0]]
        # 尝试从模板推断设备，如果失败则回退到CPU
        try:
             device = next(iter(model_template_dict.values())).device
        except StopIteration:
             device = torch.device('cpu') # Fallback
        except AttributeError:
             device = torch.device('cpu') # Fallback for non-tensor items

        honest_updates_flat = {
            cid: flatten_update(upd).to(device) for cid, upd in honest_updates.items()
        }
        honest_updates_flat_list = list(honest_updates_flat.values())

        # --- 2. 计算诚实集群的统计数据 ---
        mean_honest_flat = torch.stack(honest_updates_flat_list, dim=0).mean(dim=0)

        max_honest_dist_sq = 0.0
        if len(honest_updates_flat_list) > 1: # 确保至少有两个点
            for i, j in combinations(range(len(honest_updates_flat_list)), 2):
                dist_sq = torch.sum((honest_updates_flat_list[i] - honest_updates_flat_list[j])**2)
                if dist_sq > max_honest_dist_sq:
                    max_honest_dist_sq = dist_sq
        # 使用 item() 获取 Python 数值
        max_honest_dist = np.sqrt(max_honest_dist_sq.item())

        if max_honest_dist < 1e-9:
            print("    [攻击模拟] MinMax: 诚实梯度集群直径接近零，无法有效攻击，跳过。")
            return client_updates

        # --- 3. 初始恶意梯度方向：与均值相反 ---
        malicious_direction = -(mean_honest_flat.detach())
        norm_malicious_dir = torch.norm(malicious_direction, p=2)

        if norm_malicious_dir < 1e-9:
             print("    [攻击模拟] MinMax: 诚实梯度均值接近零，无法确定攻击方向，跳过。")
             return client_updates

        malicious_direction /= norm_malicious_dir

        # --- 4. 计算满足 beta * 直径约束的最大缩放因子 gamma ---
        # 目标边界距离
        target_max_dist = max_honest_dist * beta

        # 初始尝试：将恶意梯度放置在距离均值等于 *原始* 直径的位置
        gamma_initial = max_honest_dist
        malicious_flat_initial = mean_honest_flat.detach() + gamma_initial * malicious_direction

        # 计算初始尝试点到所有诚实点的最大距离
        current_max_dist_to_mal_sq = 0.0
        for h_flat in honest_updates_flat_list:
            dist_sq = torch.sum((malicious_flat_initial - h_flat)**2)
            if dist_sq > current_max_dist_to_mal_sq:
                current_max_dist_to_mal_sq = dist_sq
        current_max_dist_to_mal = np.sqrt(current_max_dist_to_mal_sq.item())

        # 如果初始尝试点已经超出了 *目标* 边界 (beta * 直径)
        if current_max_dist_to_mal > target_max_dist:
            # 按比例缩小 gamma，使其恰好落在目标边界上
            scale_factor = target_max_dist / (current_max_dist_to_mal + 1e-9)
            gamma_adjusted = gamma_initial * scale_factor
            malicious_flat_final = mean_honest_flat.detach() + gamma_adjusted * malicious_direction
            print(f"    [攻击模拟] MinMax: 初始尝试超出约束({current_max_dist_to_mal:.4f} > {target_max_dist:.4f})，"
                  f"已将 gamma 调整为 {gamma_adjusted:.4f}")
        else:
            # 如果初始尝试点就在目标边界内或边界上，可以直接使用
            malicious_flat_final = malicious_flat_initial
            print(f"    [攻击模拟] MinMax: 初始尝试满足约束 ({current_max_dist_to_mal:.4f} <= {target_max_dist:.4f})，"
                  f"使用 gamma = {gamma_initial:.4f}")
            # 注意：这里没有乘以 beta，因为初始的 gamma_initial 就已经是基于原始直径计算的
            # 如果 beta < 1，那么 target_max_dist 会更小，上面的 if 条件更有可能触发并进行缩减。
            # 如果 beta > 1，那么 target_max_dist 会更大，更有可能直接使用 gamma_initial。

        # --- 5. 将最终的恶意扁平向量恢复为字典结构 ---
        malicious_update_dict = unflatten_update(malicious_flat_final, model_template_dict)

        # --- 6. 替换所有恶意客户端的更新 ---
        for cid in malicious_client_ids:
            client_updates[cid] = copy.deepcopy(malicious_update_dict)

        print("    [攻击模拟]  MinMax 攻击制作并替换完成。")

        return client_updates


# --- [新版] MinSumAttack 类 (基于标准差偏离 + 可选边界约束) ---
class MinSumAttack(AttackStrategy):
    """
    实现 Min-Sum 攻击的一个变体。
    恶意更新从诚实均值 m0 出发，沿着与标准差 sigma 成比例的方向移动：
    m = m0 - gamma * sigma。
    gamma 是一个可配置参数 (MINSUM_GAMMA)，默认为 1.0。
    (可选) 还可以配置 MINSUM_APPLY_CONSTRAINT=True，以额外强制满足
    Min-Max 的 max d(m, h) <= beta * diameter 约束。
    """

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]
        honest_client_ids = list(honest_updates.keys())

        # 需要至少一个诚实客户端来计算均值/标准差
        if not honest_updates or not malicious_client_ids:
            # ... (打印跳过信息，同之前的版本) ...
            if not honest_updates:
                print("    [攻击模拟] MinSum (std_dev): 没有诚实客户端更新，无法计算统计数据，跳过攻击。")
            elif not malicious_client_ids:
                print("    [攻击模拟] MinSum (std_dev): 没有恶意客户端参与本轮。")
            return client_updates

        print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，"
              f"基于 {len(honest_updates)} 个诚实客户端，开始制作 MinSum (std_dev) 攻击...")

        # --- 0. 从 config 获取参数 ---
        gamma = float(self.config.get("MINSUM_GAMMA", 1.0))  # 控制偏离标准差的倍数，默认 1.0
        apply_constraint = bool(self.config.get("MINSUM_APPLY_CONSTRAINT", False))  # 是否应用 Min-Max 边界约束，默认 False
        beta = float(self.config.get("MINSUM_BETA", 1.0))  # 如果应用约束，使用的 beta 值，默认 1.0
        print(
            f"    [攻击模拟] MinSum (std_dev): 使用 gamma = {gamma:.2f}, apply_constraint = {apply_constraint}, beta = {beta:.2f}")

        # --- 1. 展平所有诚实更新 ---
        model_template_dict = honest_updates[honest_client_ids[0]]
        try:
            device = next(iter(model_template_dict.values())).device
        except:
            device = torch.device('cpu')  # Fallback

        honest_updates_flat = {
            cid: flatten_update(upd).to(device) for cid, upd in honest_updates.items()
        }
        honest_updates_flat_list = list(honest_updates_flat.values())

        # --- 2. 计算诚实集群的统计数据 (均值 m0 和 标准差 sigma) ---
        stacked_honest_updates = torch.stack(honest_updates_flat_list, dim=0)
        mean_honest_flat = stacked_honest_updates.mean(dim=0)  # m0

        # 计算标准差，处理只有一个样本的情况
        if stacked_honest_updates.size(0) > 1:
            std_honest_flat = stacked_honest_updates.std(dim=0)  # sigma
        else:
            std_honest_flat = torch.zeros_like(mean_honest_flat)  # sigma = 0

        # --- 3. 构造初始恶意梯度 (m = m0 - gamma * sigma) ---
        # 处理 sigma 可能为零或 NaN 的情况 (类似 LIE 中的处理)
        small_std_value = 1e-5
        # 检查 NaN 或 接近 0
        if torch.isnan(std_honest_flat).any() or torch.all(std_honest_flat.abs() < 1e-9):
            # print(f"    [攻击模拟] MinSum (std_dev): 标准差无效或为零，用小值 {small_std_value} 代替。")
            sigma_processed = torch.ones_like(mean_honest_flat) * small_std_value
        else:
            # clamp 保证数值稳定性
            sigma_processed = std_honest_flat.clamp(min=1e-6)

        malicious_flat_initial = mean_honest_flat.detach() - gamma * sigma_processed.detach()
        malicious_flat_final = malicious_flat_initial  # 默认最终结果

        # --- 4. (可选) 检查并应用 Min-Max 边界约束 ---
        if apply_constraint and len(honest_updates_flat_list) >= 2:
            # --- 计算直径 D (与 MinMax 相同) ---
            max_honest_dist_sq = 0.0
            for i, j in combinations(range(len(honest_updates_flat_list)), 2):
                dist_sq = torch.sum((honest_updates_flat_list[i] - honest_updates_flat_list[j]) ** 2)
                if dist_sq > max_honest_dist_sq:
                    max_honest_dist_sq = dist_sq
            max_honest_dist = np.sqrt(max_honest_dist_sq.item())  # D

            if max_honest_dist > 1e-9:
                target_max_dist = max_honest_dist * beta  # 目标边界

                # --- 检查初始恶意点是否超出边界 (与 MinMax 相同) ---
                current_max_dist_to_mal_sq = 0.0
                for h_flat in honest_updates_flat_list:
                    dist_sq = torch.sum((malicious_flat_initial - h_flat) ** 2)
                    if dist_sq > current_max_dist_to_mal_sq:
                        current_max_dist_to_mal_sq = dist_sq
                current_max_dist_to_mal = np.sqrt(current_max_dist_to_mal_sq.item())

                # --- 如果超出，则将其拉回到边界 (与 MinMax 不同的是起点和方向) ---
                if current_max_dist_to_mal > target_max_dist:
                    print(
                        f"    [攻击模拟] MinSum (std_dev): 初始点超出约束({current_max_dist_to_mal:.4f} > {target_max_dist:.4f})，"
                        f"正在将其拉回边界...")
                    # 我们需要找到从 m0 到 malicious_flat_initial 方向上，
                    # 恰好使得 max d(m, h) == target_max_dist 的那个点。
                    # 这通常需要迭代或求解，为了简单起见，我们使用一个近似：
                    # 将其投影回以均值为球心，半径为 R 的球面上 (R 需要估计，或直接用 MinMax 的逻辑)
                    # ---- 使用 MinMax 相似的缩放逻辑作为近似 ----
                    # 注意：这里的方向是 (malicious_flat_initial - mean_honest_flat)
                    direction_from_mean = malicious_flat_initial - mean_honest_flat
                    norm_direction = torch.norm(direction_from_mean, p=2)
                    if norm_direction > 1e-9:
                        direction_from_mean /= norm_direction

                        # 找到恰好在边界上的 gamma_adjusted (可能与 MinMax 的 gamma 不同)
                        # 这是一个简化/近似，我们假设边界点也大致在 m0 到 initial 点的连线上
                        scale_factor = target_max_dist / (current_max_dist_to_mal + 1e-9)
                        # 需要重新计算从 m0 出发的距离 gamma_adjusted
                        # (这部分逻辑与 MinMax 一致，因为都是在同一方向上缩放以满足边界)
                        gamma_for_scaling = torch.norm(malicious_flat_initial - mean_honest_flat, p=2)
                        gamma_adjusted = gamma_for_scaling * scale_factor
                        malicious_flat_final = mean_honest_flat.detach() + gamma_adjusted * direction_from_mean
                        print(f"    [攻击模拟] MinSum (std_dev): 已将点拉回边界。")
                    # else: 方向为0，无需移动，malicious_flat_final 保持为 initial (即均值)

            else:  # 直径为0，无法应用约束
                print("    [攻击模拟] MinSum (std_dev): 直径为零，无法应用边界约束。")

        # --- 5. 将最终的恶意扁平向量恢复为字典结构 ---
        malicious_update_dict = unflatten_update(malicious_flat_final, model_template_dict)

        # --- 6. 替换所有恶意客户端的更新 ---
        for cid in malicious_client_ids:
            client_updates[cid] = copy.deepcopy(malicious_update_dict)

        print("    [攻击模拟] MinSum (std_dev) 攻击制作并替换完成。")

        return client_updates


class MinSumAttack2(AttackStrategy):
    """
    改进的 Min-Sum 攻击实现（保持外部框架不变）。
    - 使用迭代梯度上升近似最大化 sum ||m - h||_2 的目标。
    - 支持可选的 Min-Max 边界约束。
    - 可选为不同恶意客户端加入小量差异以提高隐蔽性（MINSUM_VARIATION_STD）。
    配置项（self.config）支持：
      - MINSUM_GAMMA (float, default 1.0) : 基础缩放/步长因子（与 lr 联动）
      - MINSUM_STEPS (int, default 10)   : 梯度上升步数
      - MINSUM_LR (float, optional)     : 若设置则使用该学习率，否则使用 gamma/steps 作为 lr
      - MINSUM_APPLY_CONSTRAINT (bool, default False)
      - MINSUM_BETA (float, default 1.0)
      - MINSUM_VARIATION_STD (float, default 0.0) : 为每个恶意客户端添加的小正态扰动标准差
    """
    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]
        honest_client_ids = list(honest_updates.keys())

        # 需要至少一个诚实客户端来计算统计信息；需要至少一个恶意客户端才攻击
        if not honest_updates or not malicious_client_ids:
            if not honest_updates:
                print("    [攻击模拟] MinSum (iter): 没有诚实客户端更新，无法计算统计数据，跳过攻击。")
            elif not malicious_client_ids:
                print("    [攻击模拟] MinSum (iter): 没有恶意客户端参与本轮。")
            return client_updates

        print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，"
              f"基于 {len(honest_updates)} 个诚实客户端，开始 MinSum (iter) 攻击...")

        # --- 配置 & 超参 ---
        gamma = float(self.config.get("MINSUM_GAMMA", 1.0))
        steps = int(self.config.get("MINSUM_STEPS", 10))
        lr = float(self.config.get("MINSUM_LR", gamma / max(1, steps)))
        apply_constraint = bool(self.config.get("MINSUM_APPLY_CONSTRAINT", False))
        beta = float(self.config.get("MINSUM_BETA", 1.0))
        variation_std = float(self.config.get("MINSUM_VARIATION_STD", 0.0))  # 为每个恶意客户端加入微小差别
        eps = 1e-9

        print(f"    [攻击模拟] MinSum (iter): gamma={gamma:.4f}, steps={steps}, lr={lr:.6f}, "
              f"apply_constraint={apply_constraint}, beta={beta:.4f}, var_std={variation_std:.6f}")

        # --- 1. 扁平化诚实更新并迁移 device ---
        model_template_dict = honest_updates[honest_client_ids[0]]
        try:
            device = next(iter(model_template_dict.values())).device
        except Exception:
            device = torch.device('cpu')

        honest_updates_flat = {
            cid: flatten_update(upd).to(device) for cid, upd in honest_updates.items()
        }
        honest_flat_list = list(honest_updates_flat.values())
        X = torch.stack(honest_flat_list, dim=0)  # shape [n_honest, D]

        # --- 2. 初始点 m0: 诚实均值（作为起点） ---
        mean_honest_flat = X.mean(dim=0).detach()

        # --- 3. 如果只有一个诚实样本，std 方向退化，仍然能用迭代方案（但梯度退化） ---
        n_honest = X.size(0)

        # --- 4. 梯度上升：最大化 sum ||m - h|| ---
        # 梯度形式（对 m）为： sum_{h} (m - h) / (||m - h|| + eps)
        # 我们在 m 上做若干步上升，步长由 lr 控制
        m = mean_honest_flat.clone().detach()

        for t in range(steps):
            # 计算每个项的分母（距离）
            diffs = m.unsqueeze(0) - X  # [n_honest, D]
            dists = torch.norm(diffs, dim=1)  # [n_honest]
            # 避免除零
            inv_dists = 1.0 / (dists + eps)   # [n_honest]
            # 每项的向量贡献 = (m - h) / ||m - h||
            # grad = sum((m - h)/||m-h||) = sum(diffs * inv_dists.unsqueeze(1))
            grad = (diffs * inv_dists.unsqueeze(1)).sum(dim=0)  # [D]
            # 如果 grad 近似 0（例如所有 h == m），打破循环
            grad_norm = torch.norm(grad).item()
            if grad_norm < 1e-12:
                # 没有可上升的方向了
                break
            # 梯度上升一步
            m = m + lr * grad

            # 可选：每步对 m 做 Min-Max 约束投影（若 apply_constraint）
            if apply_constraint and n_honest >= 2:
                # 计算诚实集群直径 D（与 MinMax 类似）
                max_dist_sq = 0.0
                for i, j in combinations(range(n_honest), 2):
                    d_sq = torch.sum((X[i] - X[j])**2).item()
                    if d_sq > max_dist_sq:
                        max_dist_sq = d_sq
                diameter = float(np.sqrt(max_dist_sq))
                if diameter < 1e-12:
                    # 若直径退化为 0，则无法应用约束；跳过约束
                    pass
                else:
                    target_max = diameter * beta
                    # 计算 m 到所有 honest 的最大距离
                    max_to_m_sq = 0.0
                    for h in honest_flat_list:
                        dsq = torch.sum((m - h)**2).item()
                        if dsq > max_to_m_sq:
                            max_to_m_sq = dsq
                    max_to_m = float(np.sqrt(max_to_m_sq))
                    if max_to_m > target_max + 1e-12:
                        # 将 m 沿 (m - mean) 的方向缩放回使其恰好满足边界（近似投影）
                        dir_from_mean = (m - mean_honest_flat)
                        dir_norm = torch.norm(dir_from_mean).item()
                        if dir_norm > eps:
                            # 我们想找到 scale 使得 max_dist(scale) == target_max
                            # 近似方法：按比例缩放 dir_from_mean，使当前 max_to_m 缩到 target_max
                            scale = target_max / (max_to_m + eps)
                            m = mean_honest_flat + dir_from_mean * scale
                        else:
                            # 方向近似为0, 不处理
                            pass

        malicious_flat_base = m.detach()

        # --- 5. 最终化 & 为各恶意客户端生成微差别（可选） ---
        malicious_update_dict = unflatten_update(malicious_flat_base, model_template_dict)

        # If variation_std > 0, add small gaussian noise to each malicious client's update
        for cid in malicious_client_ids:
            if variation_std > 0.0:
                # create small noise vector in the flattened space, then unflatten
                noise = torch.normal(mean=0.0, std=variation_std, size=malicious_flat_base.shape, device=malicious_flat_base.device)
                mal_flat = malicious_flat_base + noise
                client_updates[cid] = copy.deepcopy(unflatten_update(mal_flat, model_template_dict))
            else:
                client_updates[cid] = copy.deepcopy(malicious_update_dict)

        print("    [攻击模拟] MinSum (iter) 攻击制作并替换完成。")
        return client_updates

# --- [新增] A Little is Enough (LIE) 协同攻击 ---
class LIEAttack(AttackStrategy):
    """
    实现 LIE 攻击。
    恶意更新被构造为 benign_mean - z_max * benign_std。
    z_max (等同于 config 中的 LIE_ATTACK_S_VALUE) 控制攻击强度。
    包含对 std 为零或 NaN 的处理。
    """
    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        # 检查是否有诚实更新和恶意客户端
        if not honest_updates or not malicious_client_ids:
            if not honest_updates:
                print("    [攻击模拟] LIE: 没有诚实客户端更新，无法计算统计数据，跳过攻击。")
            elif not malicious_client_ids:
                print("    [攻击模拟] LIE: 没有恶意客户端参与本轮。")
            return client_updates # 无法攻击，返回原样

        print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，基于 {len(honest_updates)} 个诚实客户端，开始制作新版 LIE 攻击...")

        # --- 1. 获取 z_max (等同于 s) ---
        # 我们假设 z_max 必须由 config 提供，不实现图片中的动态计算
        z_max = float(self.config.get("LIE_ATTACK_S_VALUE", 10.0)) # 从 config 读取，默认 10.0
        print(f"    [攻击模拟] LIE: 使用 z_max = {z_max:.4f}")

        # --- 2. 计算良性更新的按层平均值 (mu) 和标准差 (sigma) ---
        benign_updates_list = list(honest_updates.values())
        # 获取一个模板以确定层名和形状
        template_keys = benign_updates_list[0].keys()
        # 获取设备信息
        try:
             device = next(iter(benign_updates_list[0].values())).device
        except StopIteration:
             device = torch.device('cpu') # Fallback
        except AttributeError:
             device = torch.device('cpu') # Fallback for non-tensor items

        avg_update = {} # mu
        std_update = {} # sigma

        # 逐层计算 mu 和 sigma
        for key in template_keys:
            # 确保只处理浮点张量
            if benign_updates_list[0][key].is_floating_point():
                # 收集所有诚实客户端的该层更新，并在指定设备上堆叠
                layer_updates = torch.stack([upd[key].detach().to(device) for upd in benign_updates_list], dim=0)

                # 计算均值
                avg_update[key] = layer_updates.mean(dim=0)

                # 计算标准差，并处理只有一个诚实更新的情况 (std 会是 NaN)
                if layer_updates.size(0) > 1:
                    std_update[key] = layer_updates.std(dim=0)
                else:
                    # 如果只有一个诚实更新，标准差无意义，设为 0
                    std_update[key] = torch.zeros_like(avg_update[key])
            else:
                 # 对于非浮点类型，直接复制模板值，它们的 mu=自身, sigma=0
                 avg_update[key] = benign_updates_list[0][key].clone().to(device)
                 std_update[key] = torch.zeros_like(avg_update[key])


        # --- 3. 构造恶意更新 (malicious_update = mu - z_max * sigma) ---
        malicious_update_constructed = {}
        small_std_value = 1e-5 # 用于替换 NaN 或 0 的 sigma

        for key in template_keys:
            mu_tensor = avg_update[key]
            sigma_tensor = std_update[key]

            # --- 处理标准差 sigma ---
            if sigma_tensor is not None and sigma_tensor.is_floating_point():
                # 检查 NaN (可能发生在只有一个诚实客户端时) 或接近 0
                # torch.isnan(sigma_tensor).any() 检查张量中是否有NaN
                # torch.all(sigma_tensor.abs() < 1e-9) 检查是否所有元素都接近0
                if torch.isnan(sigma_tensor).any() or torch.all(sigma_tensor.abs() < 1e-9):
                    # print(f"    [攻击模拟] LIE: 层 {key} 的 sigma 无效或为零，用小值 {small_std_value} 替换。")
                    # 使用一个小的正值代替，避免乘法结果为0或NaN
                    sigma_tensor = torch.ones_like(mu_tensor) * small_std_value
                else:
                    # 图片中的 clamp(min=1e-6) 确保 sigma 不会太小
                    sigma_tensor = sigma_tensor.clamp(min=1e-6)

                # 计算恶意层更新
                p_mal_tensor = mu_tensor - z_max * sigma_tensor
                malicious_update_constructed[key] = p_mal_tensor

            else: # 非浮点层直接复制均值 (因为 sigma 为 0)
                malicious_update_constructed[key] = mu_tensor

        # --- 4. 替换所有恶意客户端的更新 ---
        # 不需要像图片中那样转到 CPU
        for cid in malicious_client_ids:
            client_updates[cid] = copy.deepcopy(malicious_update_constructed)

        print("    [攻击模拟] 新版 LIE 攻击 (mu - z*sigma) 制作并替换完成。")

        return client_updates


# --- [新增] 符号翻转攻击 (Sign-Flipping Attack) ---
class SignFlippingAttack(AttackStrategy):
    """
    实现符号翻转攻击。
    恶意客户端将其更新的每个参数都乘以 -1。
    这是一种非协同攻击。
    """

    def execute(self, client_updates, malicious_ids, honest_updates):
        # 筛选出恶意的客户端ID
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        if not malicious_client_ids:
            # 如果没有恶意客户端，则直接返回
            return client_updates

        print(f"    [攻击模拟] 对 {len(malicious_client_ids)} 个恶意客户端实施符号翻转攻击...")

        for cid in malicious_client_ids:
            malicious_update = client_updates[cid]

            # 遍历更新中的所有参数（例如模型的每一层）
            # 假设 malicious_update 是一个字典，键是层名，值是PyTorch张量
            for key in malicious_update:
                # 执行符号翻转 (乘以 -1)
                # 我们使用 -1.0 来确保操作是浮点数操作
                malicious_update[key] *= -1.0
                # 或者使用: malicious_update[key] = -malicious_update[key]

        print("    [攻击模拟] 符号翻转攻击完成。")
        return client_updates


# --- [新增] 噪声攻击 (Noise Attack) ---
class NoiseAttack(AttackStrategy):
    """
    实现噪声攻击。
    向恶意客户端的更新中添加高斯噪声。
    这是一种非协同攻击。
    """
    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]
        if not malicious_client_ids:
            return client_updates

        # 从config中获取噪声标准差，如果未定义则使用默认值
        noise_std = self.config.get("NOISE_ATTACK_STD", 0.1)

        print(f"    [攻击模拟] 对 {len(malicious_client_ids)} 个恶意客户端添加噪声 (std={noise_std})...")
        for cid in malicious_client_ids:
            malicious_update = client_updates[cid]
            for key in malicious_update:
                noise = torch.randn_like(malicious_update[key]) * noise_std
                malicious_update[key] += noise
        print("    [攻击模拟] 噪声攻击完成。")
        return client_updates

# --- [修改] 策略工厂 ---
# 将新的攻击策略添加到工厂中
attack_strategy_factory = {
    "backdoor": NoServerSideAttack,
    "label_flipping": NoServerSideAttack,
    "clean_label_backdoor": NoServerSideAttack,
    "min_max": MinMaxAttack,
    "lie": LIEAttack,
    "sign_flipping": SignFlippingAttack,
    "noise": NoiseAttack,
    "min_sum": MinSumAttack,
    "label_shuffling": NoServerSideAttack,
}
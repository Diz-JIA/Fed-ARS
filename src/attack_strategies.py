import torch
import copy


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


class MinMaxAttack(AttackStrategy):
    """实现Min-Max攻击的策略。"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        if honest_updates and malicious_client_ids:
            print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，开始制作Min-Max攻击...")
            # 1. 计算良性更新的平均值
            avg_honest_update = {
                key: torch.stack([upd[key] for upd in honest_updates.values()], dim=0).mean(dim=0)
                for key in honest_updates[list(honest_updates.keys())[0]]
            }

            # 2. 制作恶意更新 (与良性更新方向相反，范数相同)
            avg_norm = torch.norm(torch.cat([p.flatten() for p in avg_honest_update.values()]), p=2)
            malicious_update = {key: -avg_honest_update[key] for key in avg_honest_update}
            malicious_norm = torch.norm(torch.cat([p.flatten() for p in malicious_update.values()]), p=2)

            scale = avg_norm / (malicious_norm + 1e-9)
            for key in malicious_update:
                malicious_update[key] *= scale

            # 3. 替换所有恶意客户端的更新
            for cid in malicious_client_ids:
                client_updates[cid] = malicious_update
            print("    [攻击模拟] Min-Max攻击制作并替换完成。")

        return client_updates


# --- [新增] A Little is Enough (LIE) 协同攻击 ---
class LIEAttack(AttackStrategy):
    """
    实现 "A Little is Enough" 协同攻击。
    恶意客户端将它们的更新聚合为 benign_mean + s * benign_std，
    以微小但一致的步长偏移全局模型。
    """

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        # 协同攻击需要至少一个良性客户端来计算统计数据
        if honest_updates and malicious_client_ids:
            print(f"    [攻击模拟] 检测到 {len(malicious_client_ids)} 个恶意客户端，开始制作LIE攻击...")

            # 1. 计算良性更新的按层平均值 (mu)
            benign_updates_list = list(honest_updates.values())
            avg_update = {
                key: torch.stack([upd[key] for upd in benign_updates_list], dim=0).mean(dim=0)
                for key in benign_updates_list[0]
            }

            # 2. 计算良性更新的按层标准差 (sigma)
            std_update = {
                key: torch.stack([upd[key] for upd in benign_updates_list], dim=0).std(dim=0)
                for key in benign_updates_list[0]
            }

            # 3. 根据 mu 和 sigma 构造恶意更新
            # 这里的 s 是一个超参数，控制攻击的强度
            s = self.config.get("LIE_ATTACK_S_VALUE", 10.0)  # 您可以在config.py中定义

            malicious_update = {
                key: avg_update[key] + s * std_update[key]
                for key in avg_update
            }

            # 4. 替换所有恶意客户端的更新
            for cid in malicious_client_ids:
                client_updates[cid] = copy.deepcopy(malicious_update)
            print("    [攻击模拟] LIE攻击制作并替换完成。")

        return client_updates


# --- [新增] 符号翻转攻击 (Sign-Flipping Attack) ---
class SignFlippingAttack(AttackStrategy):
    """
    实现符号翻转攻击。
    恶意客户端将其更新的每个参数都乘以 -1。
    这是一种非协同攻击。
    """

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]
        if not malicious_client_ids:
            return client_updates

        print(f"    [攻击模拟] 对 {len(malicious_client_ids)} 个恶意客户端执行符号翻转攻击...")
        for cid in malicious_client_ids:
            malicious_update = client_updates[cid]
            for key in malicious_update:
                malicious_update[key] *= -1
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


class MinSumAttack(AttackStrategy):
    """实现Min-Sum攻击的策略。"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids_in_round = [cid for cid in client_updates if cid in malicious_ids]

        if honest_updates and malicious_client_ids_in_round:
            print(f"    [攻击模拟] 检测到 {len(malicious_client_ids_in_round)} 个恶意客户端，开始制作Min-Sum攻击...")
            # 1. 计算良性更新的平均值
            avg_honest_update = {
                key: torch.stack([upd[key] for upd in honest_updates.values()], dim=0).mean(dim=0)
                for key in honest_updates[list(honest_updates.keys())[0]]
            }

            # 2. 将所有恶意客户端的更新替换为这个“平均更新”
            for cid in malicious_client_ids_in_round:
                client_updates[cid] = avg_honest_update
            print("    [攻击模拟] Min-Sum攻击制作并替换完成。")

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
}
import torch


class AttackStrategy:
    """攻击策略的基类 (接口)"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        """执行攻击策略，返回处理后的 client_updates"""
        raise NotImplementedError


class NoServerSideAttack(AttackStrategy):
    """一个“空”策略，用于客户端攻击（如后门），服务器端无需操作。"""

    def execute(self, client_updates, malicious_ids, honest_updates):

        print(f"    [攻击模拟] 客户端侧攻击，服务器端无操作。")
        return client_updates  # 直接返回原始更新


class MinMaxAttack(AttackStrategy):
    """实现Min-Max攻击的策略。"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        if honest_updates and malicious_client_ids:
            print(f"    [攻击模拟] 服务器端开始制作Min-Max攻击...")
            # 1. 计算良性更新的平均值
            avg_honest_update = {
                key: torch.stack([upd[key] for upd in honest_updates.values()], dim=0).mean(dim=0)
                for key in honest_updates[list(honest_updates.keys())[0]]
            }

            # 2. 制作恶意更新
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

class MinSumAttack(AttackStrategy):
    """实现Min-Sum攻击的策略。"""

    def execute(self, client_updates, malicious_ids, honest_updates):
        malicious_client_ids = [cid for cid in client_updates if cid in malicious_ids]

        if honest_updates and malicious_client_ids:
            print(f"    [攻击模拟] 服务器端开始制作Min-Sum攻击...")
            # 1. 计算良性更新的平均值
            avg_honest_update = {
                key: torch.stack([upd[key] for upd in honest_updates.values()], dim=0).mean(dim=0)
                for key in honest_updates[list(honest_updates.keys())[0]]
            }

            # 2. 将所有恶意客户端的更新替换为这个“平均更新”
            for cid in malicious_client_ids:
                client_updates[cid] = avg_honest_update
            print("    [攻击模拟] Min-Sum攻击制作并替换完成。")

        return client_updates

# --- 策略工厂 ---
# 这是一个字典，将config中的字符串映射到对应的策略类
attack_strategy_factory = {
    "backdoor": NoServerSideAttack,
    "label_flipping": NoServerSideAttack,
    "clean_label_backdoor": NoServerSideAttack,
    "min_max": MinMaxAttack,
    "min_sum": MinSumAttack,
    # 未来如果增加新的服务器端攻击，只需在这里添加一行
}
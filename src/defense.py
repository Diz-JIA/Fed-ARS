import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .models import SimpleCNN # 从同目录的models.py导入SimpleCNN


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


def calculate_adversarial_vulnerability(model_state_dict, probe_images, device, config):
    """
    [新版] 通过FGSM计算模型的“对抗性脆弱度分数”。
    此版本比较的是最后一层 (Logits) 的差异。
    """
    model = SimpleCNN().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    epsilon = config["ADVERSARIAL_EPSILON"]
    scores = []

    for img in probe_images:
        img = img.unsqueeze(0).to(device)
        img.requires_grad = True

        # [优化] 使用新方法，一次前向传播同时获得输出和特征
        outputs, features_clean = model.forward_with_features(img)

        loss = F.nll_loss(outputs, outputs.max(1)[1])
        model.zero_grad()
        loss.backward()

        grad_sign = img.grad.data.sign()
        adversarial_img = torch.clamp(img + epsilon * grad_sign, -1, 1)

        with torch.no_grad():
            # --- [核心修改] ---
            # 1. 获取干净探针的最后一层输出 (Logits)
            logits_clean = model(img)

            # 2. 获取对抗样本的最后一层输出 (Logits)
            logits_adversarial = model(adversarial_img)

            # 3. 将Logits转换为概率分布
            #    使用 log_softmax 是因为 kl_div 函数的输入要求
            prob_dist_clean = F.log_softmax(logits_clean, dim=1)
            prob_dist_adversarial = F.softmax(logits_adversarial, dim=1)

            # 4. 计算KL散度作为分数
            #    KL(P || Q) 是衡量用Q近似P时的信息损失。这里我们衡量从“对抗”到“干净”的差异。
            score = F.kl_div(prob_dist_clean, prob_dist_adversarial, reduction='batchmean', log_target=True).item()
            # --- 修改结束 ---
            scores.append(score)

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


def calculate_sparsity_score(update):
    """
    计算单个模型更新的稀疏度分数 (L∞/L1 范数比)。
    分数越高，代表更新越稀疏（能量越集中），越可疑。

    Args:
        update (dict): 客户端模型更新的 state_dict。

    Returns:
        float: 稀疏度分数。
    """
    # 1. 将所有更新参数“拉平”到一个长向量中
    flat_update = torch.cat([p.flatten() for p in update.values()])

    # 2. 计算 L∞ 范数 (所有元素绝对值的最大值)
    l_inf_norm = torch.norm(flat_update, p=float('inf')).item()

    # 3. 计算 L1 范数 (所有元素绝对值之和)
    l_1_norm = torch.norm(flat_update, p=1).item()

    # 4. 计算比率作为分数，并处理分母为0的极端情况
    return (l_inf_norm / (l_1_norm + 1e-9))

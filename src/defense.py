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
            features_adversarial = model.get_penultimate_features(adversarial_img)
            score = 1 - F.cosine_similarity(features_clean, features_adversarial).item()
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
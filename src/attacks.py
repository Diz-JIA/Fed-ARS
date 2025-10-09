import torch
from torch.utils.data import Dataset
import numpy as np
import random

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


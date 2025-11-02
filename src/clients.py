# 文件: src/client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .attacks import BackdoorDataset, LabelFlippingDataset, LabelShufflingDataset
# 从我们项目内部的其他模块导入所需组件
from . import models


class Client:
    def __init__(self, client_id, local_data, is_malicious=False, config=None):
        self.client_id, self.local_data, self.is_malicious = client_id, local_data, is_malicious
        self.config = config
        self.device = self.config["DEVICE"]

        model_name = self.config.get("MODEL_NAME", "simple_cnn")
        if model_name not in models.model_factory:
            raise ValueError(f"未知的模型名称: {model_name}")

        model_constructor = models.model_factory[model_name]

        # 从 config 中获取数据集特定参数
        in_channels = self.config["in_channels"]
        num_classes = self.config["num_classes"]

        # 将参数传递给构造函数
        self.model = model_constructor(
            in_channels=in_channels,
            num_classes=num_classes
        ).to(self.device)

        if is_malicious:
            # [核心修正] 根据ATTACK_TYPE来选择攻击方式
            if self.config["ATTACK_TYPE"] == "label_flipping":
                self.local_data = LabelFlippingDataset(
                    local_data,
                    poison_ratio=self.config["POISON_RATIO"],
                    num_classes=self.config["num_classes"]
                )
            elif self.config["ATTACK_TYPE"] == "backdoor":
                # (确保您代码中已经定义了BackdoorDataset类)
                self.local_data = BackdoorDataset(
                    original_dataset=local_data,
                    poison_ratio=self.config["POISON_RATIO"],
                    trigger_size=self.config["BACKDOOR_TRIGGER_SIZE"],
                    target_label=self.config["BACKDOOR_TARGET_LABEL"]
                )
            elif self.config["ATTACK_TYPE"] == "label_shuffling":
                # (注意: 这种攻击通常会忽略 POISON_RATIO，因为它会污染整个本地数据集)
                self.local_data = LabelShufflingDataset(
                    original_dataset=local_data
                )
        self.dataloader = DataLoader(self.local_data, batch_size=self.config["BATCH_SIZE"], shuffle=True)

    def train(self, global_model_state_dict, learning_rate):
        # --- [核心修改 2] 直接使用 self.model，不再创建新模型 ---
        self.model.load_state_dict(global_model_state_dict)
        # --- 修改结束 ---

        momentum = self.config.get("MOMENTUM", 0.9)
        weight_decay = self.config.get("WEIGHT_DECAY", 5e-4)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # [修改] 差异化本地训练轮数
        epochs_to_run = self.config["LOCAL_EPOCHS"]
        if self.is_malicious:
            epochs_to_run = self.config["LOCAL_EPOCHS"] # 例如，恶意客户端的训练轮数是诚实的2倍
            # 您也可以在这里设置一个固定的更多轮数，比如 10

        self.model.train()

        # --- [新增调试代码 - Part 1] ---
        # 1. 从数据加载器中取一个批次的数据，用于前后对比
        try:
            test_batch_data, test_batch_target = next(iter(self.dataloader))
            test_batch_data, test_batch_target = test_batch_data.to(self.device), test_batch_target.to(self.device)

            # 2. 在训练前，计算这个批次的损失
            with torch.no_grad():
                pre_train_loss = nn.CrossEntropyLoss()(self.model(test_batch_data), test_batch_target)
        except StopIteration:
            print(f"    [客户端 {self.client_id} 调试] 数据加载器为空，无法进行本地训练。")
            return {key: torch.zeros_like(global_model_state_dict[key]) for key in global_model_state_dict}  # 返回零更新
        # --- [调试代码结束] ---

        for _ in range(epochs_to_run):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(data), target)
                loss.backward()
                optimizer.step()

        # --- [新增调试代码 - Part 2] ---
        # 3. 在训练后，再次计算同一个批次的损失
        with torch.no_grad():
            post_train_loss = nn.CrossEntropyLoss()(self.model(test_batch_data), test_batch_target)

        # print(f"    [客户端 {self.client_id} 调试] 训练前损失: {pre_train_loss.item():.4f} | 训练后损失: {post_train_loss.item():.4f}")

        update = {}
        new_state_dict = self.model.state_dict()
        # 获取所有可训练参数的名称集合，这样查找效率更高
        trainable_param_names = {name for name, _ in self.model.named_parameters()}

        for key in global_model_state_dict:
            if key in trainable_param_names:
                # 如果是可训练参数，计算更新增量 (delta)
                update[key] = new_state_dict[key] - global_model_state_dict[key]
            else:
                # 如果是缓冲区 (如 BatchNorm 的 running_mean/var), 直接发送新状态
                update[key] = new_state_dict[key]

        flat_update = torch.cat([p.flatten() for p in update.values()])
        update_norm = torch.norm(flat_update, p=2)
        # print(f"    [客户端 {self.client_id} 调试] 本地更新范数 (Norm): {update_norm.item():.4f}")
        # --- [调试代码结束] ---
        return update

# 文件: src/client.py

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .attacks import BackdoorDataset, LabelFlippingDataset
# 从我们项目内部的其他模块导入所需组件
from .models import SimpleCNN


class Client:
    def __init__(self, client_id, local_data, is_malicious=False, config=None):
        self.client_id, self.local_data, self.is_malicious = client_id, local_data, is_malicious
        self.config = config
        self.device = self.config["DEVICE"]
        if is_malicious:
            # [核心修正] 根据ATTACK_TYPE来选择攻击方式
            if self.config["ATTACK_TYPE"] == "label_flipping":
                self.local_data = LabelFlippingDataset(local_data, poison_ratio=self.config["POISON_RATIO"])
            elif self.config["ATTACK_TYPE"] == "backdoor":
                # (确保您代码中已经定义了BackdoorDataset类)
                self.local_data = BackdoorDataset(
                    original_dataset=local_data,
                    poison_ratio=self.config["POISON_RATIO"],
                    trigger_size=self.config["BACKDOOR_TRIGGER_SIZE"],
                    target_label=self.config["BACKDOOR_TARGET_LABEL"]
                )
        self.dataloader = DataLoader(self.local_data, batch_size=self.config["BATCH_SIZE"], shuffle=True)

    def train(self, global_model_state_dict, learning_rate):
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # [修改] 差异化本地训练轮数
        epochs_to_run = self.config["LOCAL_EPOCHS"]
        if self.is_malicious:
            epochs_to_run = self.config["LOCAL_EPOCHS"] # 例如，恶意客户端的训练轮数是诚实的2倍
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


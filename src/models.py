import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_stack = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_stack = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 512), nn.ReLU())
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        return self.classifier(self.fc_stack(self.conv_stack(x)))

    def get_penultimate_features(self, x):
        return self.fc_stack(self.conv_stack(x))

    # --- 一次性返回最终输出和特征的优化方法 ---
    def forward_with_features(self, x):
        penultimate_features = self.fc_stack(self.conv_stack(x))
        logits = self.classifier(penultimate_features)
        return logits, penultimate_features

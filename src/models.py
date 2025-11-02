import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()

        # [修改] 第一个卷积层使用 in_channels
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        # 假设输入是 32x32, 2次池化后是 8x8
        # 如果是 MNIST (28x28), 2次池化后是 7x7
        # 为了简单起见，我们使用 AdaptiveMaxPool2d 来确保输出尺寸固定为 8x8
        # ---
        # [更好的修改] 我们来计算一下:
        # CIFAR-10 (32x32) -> pool(16x16) -> pool(8x8).  fc_in = 64 * 8 * 8
        # MNIST (28x28) -> pool(14x14) -> pool(7x7).  fc_in = 64 * 7 * 7
        # 我们需要让这个也动态化

        # [最健壮的修改] 使用 AdaptiveAvgPool2d
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),  # <-- 现在输入固定为 64*8*8
            nn.ReLU()
        )

        # [修改] 最后一个分类层使用 num_classes
        self.classifier = nn.Linear(512, num_classes)

    def _run_adaptive_pool(self, x):
        """
        一个辅助函数，用于安全地运行 adaptive_pool，
        处理在 'mps' 设备上 non-divisible 尺寸调整的 Bug。
        """
        # 检查张量是否在 'mps' 设备上
        if x.device.type == 'mps':
            # 1. 移到 CPU
            # 2. 在 CPU 上执行池化
            # 3. 移回原始的 'mps' 设备
            return self.adaptive_pool(x.cpu()).to(x.device)
        else:
            #
            # 对于 'cpu' 或 'cuda' (Nvidia)，直接执行
            return self.adaptive_pool(x)

    def forward(self, x):
        x_conv = self.conv_stack(x)
        x_pooled = self._run_adaptive_pool(x_conv)  # <-- [修正] 添加此行
        x_fc = self.fc_stack(x_pooled)
        return self.classifier(x_fc)

    def get_penultimate_features(self, x):
        x_conv = self.conv_stack(x)
        x_pooled = self._run_adaptive_pool(x_conv) # <-- [修正] 添加此行
        return self.fc_stack(x_pooled)

    def forward_with_features(self, x):
        x_conv = self.conv_stack(x)
        x_pooled = self._run_adaptive_pool(x_conv) # <-- [修正] 添加此行
        penultimate_features = self.fc_stack(x_pooled)
        logits = self.classifier(penultimate_features)
        return logits, penultimate_features



# --- [新增] 创建ResNet-18模型的函数 ---
def create_resnet18_for_cifar(in_channels=3, num_classes=10):
    model = models.resnet18(weights=None)

    # [修改] 第一个卷积层使用 in_channels
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    num_ftrs = model.fc.in_features
    # [修改] 最后一个分类层使用 num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

model_factory = {
    "simple_cnn": SimpleCNN,
    "resnet18": create_resnet18_for_cifar,
    # 如果您未来添加了 MobileNetV2, 也可以在这里注册:
    # "mobilenet_v2": create_mobilenet_v2_for_cifar,
}

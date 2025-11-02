import torch

# 可选项: "cifar10", "mnist"
DATASET_NAME = "cifar10"

# 数据集专属配置字典
# 在这里定义每个数据集的独特属性
# 这样可以确保模型、加载器和服务器自动获取正确参数
DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "data_mean": (0.4914, 0.4822, 0.4465),
        "data_std": (0.2023, 0.1994, 0.2010),
        "probe_file": "data/clustered_probe_indices_for_cifar10.npy"
    },
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,  # MNIST 是 1 通道
        "data_mean": (0.1307,),
        "data_std": (0.3081,),
        "probe_file": "data/clustered_probe_indices_for_mnist.npy"
    }
}

# 通用配置字典
base_config = {
    # 实验环境设置
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "mps"),
    "SEED": 123,

    # 实验日志文件路径
    "LOG_FILE_PATH": "results/logs/defense_config/scene3.csv",

    # 联邦学习设置
    "NUM_CLIENTS": 100,  # 增加客户端总数以更好地模拟统计检测
    # --- [新增] 数据分区设置 ---
    "DATA_PARTITION_STRATEGY": "dirichlet", # 可选项: "iid", "dirichlet"
    "DIRICHLET_ALPHA": 0.5,               # Dirichlet 分布参数 (alpha 越小, Non-IID 程度越高)
    "FL_ROUNDS": 130,
    "CLIENTS_PER_ROUND": 10,  # 每轮随机挑选10个客户端参与
    "MODEL_NAME":"simple_cnn",  # 默认为simple_cnn
    # 客户端本地训练参数
    "LEARNING_RATE": 0.01,  # simple_cnn+cifar10设置为 0.005，simple_cnn+mnist设置为 0.01,resnet设置为 0.001
    "LOCAL_EPOCHS": 3,  # simple_cnn设置为 3，resnet设置为 5
    "BATCH_SIZE": 64,
    # 学习率调度器 (Scheduler) 设置
    "SCHEDULER_ENABLED": True,
    # "SCHEDULER_PATIENCE": 5,   # 连续5轮 test_loss 不下降就调整
    # "SCHEDULER_FACTOR": 0.5,   # 学习率衰减为原来的一半 (lr = lr * 0.5)
    "SCHEDULER_TYPE": "multistep",  # <--- [新增] 指定调度器类型
    "SCHEDULER_MILESTONES": [80],  # <--- 对于 simplecnn设置为【20】
    "SCHEDULER_GAMMA": 0.1,  # <--- 每次降低学习率时，乘以0.1
    # 是否早停
    "EARLY_STOPPING_ENABLED": False,  # True启用早停
    "EARLY_STOPPING_PATIENCE": 15,  # 连续10轮性能不提升就停止
    "EARLY_STOPPING_METRIC": "loss",  # 监控指标: 'loss' 或 'accuracy'

    # 攻击设置
    "ATTACK_TYPE": "lie",
    "MALICIOUS_CLIENTS": 20,
    "POISON_RATIO": 0.7,
    "BACKDOOR_TRIGGER_SIZE": 5,
    "BACKDOOR_TARGET_LABEL": 0,
    "LIE_ATTACK_S_VALUE": 1.5, # LIE攻击的强度参数
    "NOISE_ATTACK_STD": 0.10, # 噪声攻击的标准差
    # beta=1.0: 攻击者试图精确地落在诚实集群的边界上
    # beta<1.0: 攻击者更保守，落在边界内部，可能更隐蔽
    # beta>1.0: 攻击者更激进，试图超出边界 (但仍会被缩放回边界)
    "MINMAX_BETA":1.0, # min_max攻击的强度参数
    # [新增] MinSum (std_dev) 攻击参数
    "MINSUM_GAMMA": 1.0,                  # 设置偏离 1 倍标准差
    "MINSUM_APPLY_CONSTRAINT": True,     # 强制执行 Min-Max 边界约束
    "MINSUM_BETA": 1.0,                  # 边界约束的 beta 值 (仅当上面为 True 时有效)

    # 优化器参数
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 5e-4,

    # 聚合方法
    "AGGREGATION_METHOD": "avg",  # 可选项: "avg", "trimmed_mean","krum"
    "TRIMMED_MEAN_BETA": 0.2,              # 裁剪比例,按照原论文设置为恶意客户端比例
    "KRUM_F": 4,  # 假设每轮聚合最多的拜占庭客户端数量 (f)
    "KRUM_M": 1,   # 要选择的客户端数量 (m=1 是标准 Krum)


    # 防御设置
    "DEFENSE_ENABLED": True,  # 是否启用在线防御
    "DEFENSE_START_ROUND": 1,  # 从第5轮开始执行防御，给模型一点初始收敛时间

    # 声誉与降权模块参数
    "REPUTATION_THRESHOLD":1,
    "REPUTATION_DECAY_FACTOR": 0.5, # γ值 (gamma)

    # 双重审查模型参数
    "ADVERSARIAL_EPSILON": 0.05,  # FGSM扰动大小，一个常用的值
    "VULNERABILITY_MAD_THRESHOLD": 1.2,  # 对抗脆弱度 MAD 阈值，值越低越防御系统越敏感
    # 动态阈值增长因子 (K)
    # 1.0 = 原版 (随客户端减少, 阈值最高增长到 2*base)
    # 2.0 = 增长更快 (随客户端减少, 阈值最高增长到 3*base)
    "DYNAMIC_THRESHOLD_K": 6.0,
    "CLIP_MAX_NORM":1.2,  #裁剪阈值


    "DISTANCE_MAD_THRESHOLD": 2,  # 值越低越防御系统越敏感
    "PERTURBATION_STRENGTH": 0.1,  # 扰动强度
    "SIMILARITY_LOW_MAD_THRESHOLD": 2.0, # 相似度声望分的“低分”阈值 (越小越宽松)
    "IDA_LOW_MAD_THRESHOLD": 2.0,         # IDA不稳定性增量的“低分”阈值 (越小越宽松)




}


# 将选择的数据集配置合并到主配置中
def get_config():
    config = base_config.copy()
    dataset_cfg = DATASET_CONFIGS.get(DATASET_NAME)

    if dataset_cfg is None:
        raise ValueError(f"未知的 DATASET_NAME: {DATASET_NAME}")

    # 将数据集特定配置（如 num_classes, in_channels）添加到主 config
    config.update(dataset_cfg)
    # 也保留原始字典，方便调用
    config["DATASET_NAME"] = DATASET_NAME

    return config


# 最终导出的配置
config = get_config()
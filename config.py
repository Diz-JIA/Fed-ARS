import torch

config = {
    # 实验环境设置
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "mps"),
    "SEED": 123,

    # 实验日志文件路径
    "LOG_FILE_PATH": "results/logs/ttr_simulation/scene3.csv",

    # 联邦学习设置
    "NUM_CLIENTS": 20,  # 增加客户端总数以更好地模拟统计检测
    "FL_ROUNDS": 200,  # 模拟的总轮数
    "CLIENTS_PER_ROUND": 10,  # 每轮随机挑选10个客户端参与

    # 客户端本地训练参数
    "LOCAL_EPOCHS": 3,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.01,

    # 攻击设置
    "ATTACK_TYPE": "min_max",
    "MALICIOUS_CLIENTS": 8,
    "POISON_RATIO": 0.7,
    "BACKDOOR_TRIGGER_SIZE": 5,
    "BACKDOOR_TARGET_LABEL": 0,

    # 防御设置
    "DEFENSE_ENABLED": True,  # 是否启用在线防御
    "DEFENSE_START_ROUND": 5,  # 从第5轮开始执行防御，给模型一点初始收敛时间

    # 双重审查模型参数
    "ADVERSARIAL_EPSILON": 0.05,  # FGSM扰动大小，一个常用的值
    "VULNERABILITY_MAD_THRESHOLD": 3.0,  # 值越高越严格

    "DISTANCE_MAD_THRESHOLD": 2.5,  # 值越高越严格

    "PERTURBATION_STRENGTH": 0.1,  # 扰动强度
    "SIMILARITY_LOW_MAD_THRESHOLD": 2.0, # 相似度声望分的“低分”阈值 (越小越宽松)
    "IDA_LOW_MAD_THRESHOLD": 2.0,         # IDA不稳定性增量的“低分”阈值 (越小越宽松)

    "CLIP_MAX_NORM":1.2,  #裁剪阈值

    # 声誉与降权模块参数

    "REPUTATION_DECAY_FACTOR": 0.5, # γ值 (gamma)


    # 学习率调度器 (Scheduler) 设置
    "SCHEDULER_ENABLED": True,
    "SCHEDULER_PATIENCE": 5,   # 连续5轮 test_loss 不下降就调整
    "SCHEDULER_FACTOR": 0.5,   # 学习率衰减为原来的一半 (lr = lr * 0.5)

    # 是否早停
    "EARLY_STOPPING_ENABLED": True,      # True启用早停
    "EARLY_STOPPING_PATIENCE": 10,       # 连续10轮性能不提升就停止
    "EARLY_STOPPING_METRIC": "loss",     # 监控指标: 'loss' 或 'accuracy'

}
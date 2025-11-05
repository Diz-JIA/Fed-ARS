from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np


def get_data(config):
    """
    根据 config 动态加载指定的数据集 (CIFAR-10 或 MNIST)。
    """
    dataset_name = config["DATASET_NAME"]
    print(f"    [数据加载器] 正在加载 {dataset_name.upper()} 数据集...")

    # 从 config 中获取数据集专属的均值和方差
    data_mean = config["data_mean"]
    data_std = config["data_std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


    return train_dataset, test_dataset


def create_iid_partitions(dataset, config):
    """
    [修改后] 创建 IID 数据分区，从 config 获取参数。
    """
    num_clients = config["NUM_CLIENTS"]  # <--- [修改]
    print(f"    [数据加载器] 正在创建 {num_clients} 个 IID 分区...")

    num_items = len(dataset) // num_clients
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    print("    [数据加载器] IID 分区创建完毕。")
    return [Subset(dataset, list(idxs)) for _, idxs in dict_users.items()]


def create_dirichlet_partitions(dataset, config):
    """
    使用 Dirichlet 分布创建 Non-IID 数据分区。
    Alpha 越小，Non-IID 程度越高。
    """
    num_clients = config["NUM_CLIENTS"]
    alpha = config["DIRICHLET_ALPHA"]
    num_classes = config["num_classes"]

    print(f"    [数据加载器] 正在创建 {num_clients} 个 Dirichlet (Non-IID) 分区 (alpha={alpha})...")

    # 1. 获取数据集的所有标签
    try:
        # 适用于 MNIST, CIFAR-10 等标准 torchvision 数据集
        labels = np.array(dataset.targets)
    except AttributeError:
        # 如果 .targets 属性不存在，尝试迭代
        print("    [警告] .targets 属性不存在，将迭代数据集获取标签。这可能较慢。")
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 2. 按类别组织样本索引
    indices_by_label = [np.where(labels == i)[0] for i in range(num_classes)]

    # 3. 为每个类别生成狄利克雷分布的客户端比例
    # shape: [num_classes, num_clients]
    client_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)

    # 4. 初始化每个客户端的索引列表
    client_indices = [[] for _ in range(num_clients)]

    # 5. 分配索引
    for k in range(num_classes):
        label_k_indices = indices_by_label[k]
        np.random.shuffle(label_k_indices)  # 确保随机分配样本

        proportions_k = client_proportions[k, :]
        samples_per_client_k = (proportions_k * len(label_k_indices)).astype(int)

        # 确保所有样本被分配 (处理舍入误差)
        remainder = len(label_k_indices) - samples_per_client_k.sum()
        if remainder > 0:
            # 将余数随机(按比例)分配给客户端
            add_indices = np.random.choice(num_clients, remainder, replace=True, p=proportions_k / proportions_k.sum())
            for i in add_indices:
                samples_per_client_k[i] += 1

        # 计算分割点
        split_points = np.cumsum(samples_per_client_k).astype(int)[:-1]

        # 分割索引
        client_splits_for_class_k = np.split(label_k_indices, split_points)

        # 将分割后的索引添加到各自的客户端列表中
        for j in range(num_clients):
            client_indices[j].extend(client_splits_for_class_k[j])

    print(f"    [数据加载器] Dirichlet (alpha={alpha}) 分区创建完毕。")
    min_len = min(len(c) for c in client_indices)
    mean_len = np.mean([len(c) for c in client_indices])
    max_len = max(len(c) for c in client_indices)
    print(f"    [数据加载器] 客户端数据量 (min, mean, max): {min_len:.0f}, {mean_len:.0f}, {max_len:.0f}")

    # 6. 创建 Subsets (使用 map(int, ...) 确保索引是标准 int)
    return [Subset(dataset, list(map(int, idxs))) for idxs in client_indices]

# --- [新增] 数据分区策略工厂 ---
partition_strategy_factory = {
    "iid": create_iid_partitions,
    "dirichlet": create_dirichlet_partitions
}
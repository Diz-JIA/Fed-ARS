import torch


def trimmed_mean_aggregate(client_updates, config):
    """
    使用 Trimmed Mean 鲁棒聚合算法来聚合客户端更新。
    这是一个独立的函数，可以从外部调用。

    Args:
        client_updates (dict): 包含本轮所有客户端更新的字典 {client_id: update_dict}。
        config (dict): 包含超参数的全局配置字典。

    Returns:
        dict: 聚合后的平均更新。
    """
    if not client_updates:
        return None

    # 从配置中获取要裁剪的比例 beta，默认值为 0.1
    beta = config.get("TRIMMED_MEAN_BETA", 0.1)
    num_clients_in_round = len(client_updates)

    # 计算每边要裁剪掉的客户端数量
    num_to_trim = int(num_clients_in_round * beta)

    # 确保不会裁剪掉所有客户端
    if 2 * num_to_trim >= num_clients_in_round:
        print(f"    [警告] Trimmed Mean 的 beta 值 ({beta}) 过高，将裁剪掉所有客户端。跳过本轮聚合。")
        return None

    print(f"    [聚合] 使用 Trimmed Mean (beta={beta})，每边裁剪 {num_to_trim} 个客户端。")

    sample_update = list(client_updates.values())[0]
    avg_update = {}

    for key in sample_update.keys():
        # 1. 获取该层参数的原始形状，用于后续恢复
        original_shape = sample_update[key].shape

        # 2. 将所有客户端在这一层的更新堆叠起来
        stacked_updates = torch.stack([client_updates[cid][key] for cid in client_updates.keys()], dim=0)

        # 3. [修复] 展平 (Reshape): 将 [num_clients, D1, D2, ...] 变为 [num_clients, N]
        #    这是解决 Apple Metal (MPS) 后端不支持 5D 张量排序的关键
        flat_updates = stacked_updates.view(num_clients_in_round, -1)

        # 4. 沿着客户端维度 (dim=0) 对【展平后】的值进行排序
        sorted_flat_updates, _ = torch.sort(flat_updates, dim=0)

        # 5. 裁剪掉最高和最低的值
        if num_to_trim > 0:
            trimmed_flat_updates = sorted_flat_updates[num_to_trim: -num_to_trim]
        else:
            trimmed_flat_updates = sorted_flat_updates  #

        # 6. 对剩余的值求平均 (结果是 1D 向量)
        avg_flat_update = torch.mean(trimmed_flat_updates, dim=0)

        # 7. [修复] 将 1D 向量恢复 (Reshape back) 为原始参数形状
        avg_update[key] = avg_flat_update.view(original_shape)

    return avg_update


def krum_aggregate(client_updates, config):
    """
    使用 (Multi-)Krum 鲁棒聚合算法来聚合客户端更新。
    该算法选择 m 个分数最低的客户端，并返回它们的平均值。
    标准 Krum 是 m=1 的特例。

    Args:
        client_updates (dict): 包含本轮所有客户端更新的字典 {client_id: update_dict}。
        config (dict): 包含超参数的全局配置字典。
                       - "KRUM_F": 假设的拜占庭/恶意客户端数量 (f)
                       - "KRUM_M": 要选择并聚合的客户端数量 (m)，默认为 1

    Returns:
        dict: 聚合后的更新（m 个被选中客户端的平均值）。
    """
    if not client_updates:
        return None

    client_ids = list(client_updates.keys())
    num_clients_in_round = len(client_ids)

    # 1. 从配置中获取超参数
    # f: 假设的恶意客户端数量
    f = config.get("KRUM_F", 0)
    # m: 要选择的客户端数量 (m=1 对应标准 Krum)
    m = config.get("KRUM_M", 1)

    # 2. 检查 Krum 的先决条件
    # Krum 算法的鲁棒性保证通常需要 n >= 2f + m
    if num_clients_in_round < 2 * f + m:
        print(
            f"    [警告] Krum (f={f}, m={m}) 需要 n >= 2f + m (即 {2 * f + m} 个客户端)，但本轮只有 {num_clients_in_round} 个。跳过聚合。")
        return None

    # k 是计算分数时要考虑的邻居数量 (n-f-m)
    k = num_clients_in_round - f - m
    if k < 0:
        # 理论上，上面的检查已经覆盖了 k < 0 的情况，但作为双重保障
        print(f"    [警告] Krum (f={f}, m={m}) 导致邻居数 k={k} < 0。跳过聚合。")
        return None

    print(f"    [聚合] 使用 Krum (f={f}, m={m})，从 {num_clients_in_round} 个客户端中选择 {m} 个。")

    # 3. 将每个客户端的所有层更新展平为一个长向量
    flattened_updates = []
    for cid in client_ids:
        # 将一个客户端的所有层张量连接成一个向量
        flat_vec = torch.cat([client_updates[cid][key].view(-1) for key in client_updates[cid].keys()])
        flattened_updates.append(flat_vec)

    # 堆叠所有客户端的展平向量 (shape: [num_clients, total_params])
    stacked_flat_updates = torch.stack(flattened_updates, dim=0)

    # 4. 计算所有客户端之间的成对欧几里得距离的平方
    # cdist 计算 p=2 (欧氏距离)，然后我们取平方
    # dists_sq 的 shape: [num_clients, num_clients]
    dists_sq = torch.cdist(stacked_flat_updates, stacked_flat_updates, p=2).pow(2)

    # 5. 计算每个客户端的分数 (score)
    # 分数是到 k 个最近邻居（不包括自己）的距离平方和
    # (k = n - f - m)

    # 对每个客户端的距离进行排序 (升序)
    sorted_dists_sq, _ = torch.sort(dists_sq, dim=1)

    # 选取从索引 1 (跳过自己) 开始的 k 个最近邻居的距离，并求和
    # sorted_dists_sq[:, 1:k+1] 包含了 k 个最小的非零距离
    scores = torch.sum(sorted_dists_sq[:, 1:k + 1], dim=1)

    # 6. 选出 m 个分数最低的客户端
    # topk (largest=False) 返回最小的 m 个值及其索引
    _, top_m_indices = torch.topk(scores, m, largest=False)

    chosen_client_ids = [client_ids[i] for i in top_m_indices.tolist()]

    if m == 1:
        print(f"    [Krum] 选中的客户端: {chosen_client_ids[0]} (分数: {scores[top_m_indices[0]]:.4f})")
    else:
        print(f"    [Krum] 选中的 {m} 个客户端: {chosen_client_ids}")

    # 7. 聚合 (对选中的 m 个客户端的更新求平均)
    # (如果 m=1，这只是返回被选中的那个客户端的更新)

    sample_update = list(client_updates.values())[0]
    aggregated_update = {}

    for key in sample_update.keys():
        # 1. 将所有被选中客户端在这一层的更新堆叠起来
        if m == 1:
            # 特殊情况：m=1，无需堆叠和平均
            aggregated_update[key] = client_updates[chosen_client_ids[0]][key].clone()
        else:
            stacked_updates = torch.stack([client_updates[cid][key] for cid in chosen_client_ids], dim=0)
            # 2. 对 m 个更新求平均
            aggregated_update[key] = torch.mean(stacked_updates, dim=0)

    return aggregated_update

# --- 聚合算法工厂 ---
# 未来您可以将 Krum, Median 等其他算法也作为函数添加到这个文件中，并在这里注册
aggregation_factory = {
    "trimmed_mean": trimmed_mean_aggregate,
    "krum": krum_aggregate, # 未来可以这样添加
}
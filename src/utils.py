# -*- coding: utf-8 -*-
"""
@Description: 存放通用辅助工具 (例如 set_seed, 日志记录, 绘图函数)

"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import csv
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # 计算损失
            loss = nn.CrossEntropyLoss()(outputs, target)
            total_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / total  # 计算平均损失
    accuracy = correct / total
    return avg_loss, accuracy  # 返回两个值


def log_experiment_results(log_file, scenario_name, config_params, metrics):
    """
    将单次实验场景的配置和结果记录到CSV文件中。
    """
    # 将场景名、配置和结果合并到一个字典中
    log_data = {"scenario": scenario_name, **config_params, **metrics}

    # 定义表头顺序 (动态生成以包含所有可能的键)
    header = ["scenario"] + list(config_params.keys()) + list(metrics.keys())

    # 检查文件是否存在
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()

            writer.writerow(log_data)

    except IOError as e:
        print(f"[错误] 无法写入日志文件: {log_file}, 原因: {e}")


def plot_results(history_dict):
    """
    根据传入的历史记录字典，绘制准确率和损失值的对比图表。

    Args:
        history_dict (dict): 一个字典，键是场景名(e.g., 'no_attack', 'under_attack', 'with_defense'),
                             值是对应的从run_simulation返回的history对象。
    """

    # --- 图表一：准确率对比 ---
    plt.figure(figsize=(10, 6))
    max_rounds_acc = 0

    # 使用 .get() 安全地获取history对象，如果某个场景没跑，会返回None
    history_no_attack = history_dict.get("no_attack")
    history_under_attack = history_dict.get("under_attack")
    history_with_defense = history_dict.get("with_defense")

    if history_no_attack:
        rounds = len(history_no_attack['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_no_attack['accuracy'], 'g-s',
                 label=f'Scenario 1: No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_under_attack['accuracy'], 'r-x',
                 label=f'Scenario 2: Under Attack (Ran for {rounds} rounds)')

    if history_with_defense:
        rounds = len(history_with_defense['accuracy'])
        max_rounds_acc = max(max_rounds_acc, rounds)
        plt.plot(range(1, rounds + 1), history_with_defense['accuracy'], 'b-o',
                 label=f'Scenario 3: With Defense (Ran for {rounds} rounds)')

    if max_rounds_acc > 0:
        plt.title("Performance Comparison (Accuracy)")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlim(0, max_rounds_acc + 5)
        plt.show()
    else:
        print("\n[可视化] 没有可用于绘制准确率图的实验结果。")

    # --- 图表二：损失值对比 ---
    plt.figure(figsize=(12, 7))
    max_rounds_loss = 0

    if history_no_attack:
        rounds = len(history_no_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_no_attack['loss'], 'g-s',
                 label=f'Scenario 1: No Attack (Converged in {rounds} rounds)')

    if history_under_attack:
        rounds = len(history_under_attack['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_under_attack['loss'], 'r-x',
                 label=f'Scenario 2: Under Attack (Ran for {rounds} rounds)')

    if history_with_defense:
        rounds = len(history_with_defense['loss'])
        max_rounds_loss = max(max_rounds_loss, rounds)
        plt.plot(range(1, rounds + 1), history_with_defense['loss'], 'b-o',
                 label=f'Scenario 3: With Defense (Ran for {rounds} rounds)')

    if max_rounds_loss > 0:
        plt.title("Performance Comparison (Loss)")
        plt.xlabel("Federated Learning Round")
        plt.ylabel("Global Model Test Loss")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.xlim(0, max_rounds_loss + 5)
        plt.show()
    else:
        print("\n[可视化] 没有可用于绘制损失图的实验结果。")
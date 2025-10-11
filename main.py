import argparse
import copy

from torch.utils.data import DataLoader

# 从我们创建的模块中导入所有需要的组件
from config import config
from src.attacks import evaluate_backdoor_asr
from src.clients import Client
from src.data_loader import get_cifar10_data, create_iid_partitions
from src.server import Server
from src.utils import set_seed, log_experiment_results, plot_results


def run_experiment():
    parser = argparse.ArgumentParser(description="选择性运行联邦学习模拟场景。")
    parser.add_argument(
        '--scenarios', '-s',
        nargs='+',  # 允许多个值
        type=int,
        default=[1, 2, 3],  # 如果不指定，则默认运行所有场景
        choices=[1, 2, 3],
        help="指定要运行的场景编号列表。1:无攻击, 2:有攻击无防御, 3:有攻击有防御。例如: --scenarios 1 3或-s 1 3"
    )
    args = parser.parse_args()
    scenarios_to_run = args.scenarios
    # print(f"[实验配置] IDA阈值 = {config["OUTLIER_THRESHOLD"]},相似度阈值 = {config["SIMILARITY_THRESHOLD"]},声誉容忍度 = {config["REPUTATION_THRESHOLD"]},扰动强度 = {config["PERTURBATION_STRENGTH"]}")

    set_seed(config["SEED"])

    # 1. 准备全局数据和客户端池
    train_dataset, test_dataset = get_cifar10_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_datasets = create_iid_partitions(train_dataset, config["NUM_CLIENTS"])

    # 定义需要记录到日志的通用参数
    common_params_to_log = {
        "LOCAL_EPOCHS": config["LOCAL_EPOCHS"],
        "LEARNING_RATE": config["LEARNING_RATE"],
    }

    # [修改] 2. 初始化所有历史记录变量，以防某些场景被跳过
    history_no_attack = None
    history_under_attack = None
    history_with_defense = None

    # [修改] 3. 使用 "if" 语句包裹每个场景
    if 1 in scenarios_to_run:
        # --- 场景一: 无攻击 ---
        print("\n\n=============== 场景一: 无攻击环境 ===============")

        config_no_attack = copy.deepcopy(config)
        config_no_attack["MALICIOUS_CLIENTS"] = 0
        config_no_attack["DEFENSE_ENABLED"] = False
        config_no_attack["SCENARIO_NAME"] = "no_attack"

        clients_no_attack = [Client(i, client_datasets[i], is_malicious=False,config=config_no_attack) for i in range(config["NUM_CLIENTS"])]
        # 将更新后的config传入Server
        server_no_attack = Server(clients_no_attack, test_loader, config_no_attack)
        history_no_attack = server_no_attack.run_simulation()

        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_no_attack.global_model, test_loader, config["DEVICE"],
                                          config_no_attack)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_1 = common_params_to_log.copy()
        # 场景一没有攻击和防御，用 N/A 覆盖
        params_log_1.update({
            "MALICIOUS_CLIENTS": "N/A",
            "POISON_RATIO": "N/A",
            "PERTURBATION_STRENGTH": "N/A",
            "SIMILARITY_LOW_MAD_THRESHOLD": "N/A",
            "IDA_LOW_MAD_THRESHOLD": "N/A",

            "REPUTATION_DECAY_FACTOR": "N/A"
        })
        actual_rounds_1 = len(history_no_attack['accuracy'])
        metrics_log_1 = {
            "actual_rounds": actual_rounds_1,
            "final_accuracy": f"{history_no_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A",
            "final_asr": f"{final_asr:.2f}%",
        }
        log_experiment_results(config["LOG_FILE_PATH"], "1", params_log_1, metrics_log_1)

    if 2 in scenarios_to_run:
        # --- 场景二: 有攻击, 无防御 ---
        print("\n\n=============== 场景二: 有攻击, 无防御 ===============")

        config_under_attack = copy.deepcopy(config)
        config_under_attack["DEFENSE_ENABLED"] = False
        config_under_attack["SCENARIO_NAME"] = "under_attack_no_defense"

        clients_under_attack = [
            Client(i, client_datasets[i], is_malicious=(i < config_under_attack["MALICIOUS_CLIENTS"]), config=config_under_attack)
            for i in range(config["NUM_CLIENTS"])]
        # [新增] 打印本场景的恶意客户端ID列表
        malicious_ids_2 = [c.client_id for c in clients_under_attack if c.is_malicious]
        print(f"    [实验设置] 本场景恶意客户端ID: {malicious_ids_2}")

        server_under_attack = Server(clients_under_attack, test_loader, config_under_attack)
        history_under_attack = server_under_attack.run_simulation()

        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_under_attack.global_model, test_loader, config["DEVICE"],
                                          config_under_attack)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_2 = common_params_to_log.copy()
        params_log_2.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": "N/A",
            "SIMILARITY_LOW_MAD_THRESHOLD": "N/A",
            "IDA_LOW_MAD_THRESHOLD": "N/A",

            "REPUTATION_DECAY_FACTOR": "N/A",
        })
        actual_rounds_2 = len(history_under_attack['accuracy'])
        metrics_log_2 = {
            "actual_rounds": actual_rounds_2,
            "final_accuracy": f"{history_under_attack['accuracy'][-1] * 100:.2f}%",
            "malicious_removed": "N/A",
            "benign_removed": "N/A",
            "final_asr": f"{final_asr:.2f}%"
        }
        log_experiment_results(config["LOG_FILE_PATH"], "2", params_log_2, metrics_log_2)

    if 3 in scenarios_to_run:
        # --- 场景三: 有攻击, 有您的IDA防御 ---
        print("\n\n=============== 场景三: 有攻击, 启用IDA防御 ===============")
        print(f"    [攻击设置] {config["ATTACK_TYPE"]}")

        config_with_defense = copy.deepcopy(config)
        config_with_defense["DEFENSE_ENABLED"] = True
        config_with_defense["SCENARIO_NAME"] = "with_defense"

        clients_with_defense = [
            Client(i, client_datasets[i], is_malicious=(i < config_with_defense["MALICIOUS_CLIENTS"]),config = config_with_defense)
            for i in range(config["NUM_CLIENTS"])]

        # [新增] 打印本场景的恶意客户端ID列表
        malicious_ids_3 = [c.client_id for c in clients_with_defense if c.is_malicious]
        print(f"    [实验设置] 本场景恶意客户端ID: {malicious_ids_3}")

        server_with_defense = Server(clients_with_defense, test_loader, config_with_defense)
        history_with_defense = server_with_defense.run_simulation()

        # [新增] 计算最终的ASR
        final_asr = evaluate_backdoor_asr(server_with_defense.global_model, test_loader, config["DEVICE"],
                                          config_with_defense)
        print(f"    [评估] 最终攻击成功率 (ASR): {final_asr:.2f}%")

        ### 修改 ###: 复用通用参数，并添加场景特定参数
        params_log_3 = common_params_to_log.copy()
        params_log_3.update({
            "MALICIOUS_CLIENTS": config["MALICIOUS_CLIENTS"],
            "POISON_RATIO": config["POISON_RATIO"],
            "PERTURBATION_STRENGTH": config["PERTURBATION_STRENGTH"],
            "SIMILARITY_LOW_MAD_THRESHOLD": config["SIMILARITY_LOW_MAD_THRESHOLD"],
            "IDA_LOW_MAD_THRESHOLD": config.get("IDA_LOW_MAD_THRESHOLD", "N/A"),

            "REPUTATION_DECAY_FACTOR": config.get("REPUTATION_DECAY_FACTOR", "N/A")
        })
        actual_rounds_3 = len(history_with_defense['accuracy'])
        # [修改] 2. 正确地统计并记录最终被“降权”的客户端数量
        # 对于“指数衰减”策略，我们定义“被降权”为最终声誉分 > 0 的客户端
        final_reputations = server_with_defense.reputation_scores
        convicted_clients = {cid for cid, rep in final_reputations.items() if rep > 0}

        malicious_clients_config = config_with_defense["MALICIOUS_CLIENTS"]
        downweighted_malicious_count = len([cid for cid in convicted_clients if cid < malicious_clients_config])
        downweighted_benign_count = len([cid for cid in convicted_clients if cid >= malicious_clients_config])

        metrics_log_3 = {
            "actual_rounds": actual_rounds_3,
            "final_accuracy": f"{history_with_defense['accuracy'][-1] * 100:.2f}%",
            "downweighted_malicious": downweighted_malicious_count,
            "downweighted_benign": downweighted_benign_count,
            "final_asr": f"{final_asr:.2f}%"
        }
        log_experiment_results(config["LOG_FILE_PATH"], "3", params_log_3, metrics_log_3)

    # print(f"\n[日志] 所有场景已成功记录到: {config['LOG_FILE_PATH']}")

    # --- [核心修改] 结果可视化 ---

    # 1. 汇总所有可能存在的history记录
    all_histories = {
        "no_attack": history_no_attack,
        "under_attack": history_under_attack,
        "with_defense": history_with_defense,
    }

    # 2. 调用封装好的绘图函数，传入字典
    plot_results(all_histories)

    # 3. (可选) 保留与server对象相关的打印逻辑
    #    因为server_with_defense只在主文件中存在，所以这部分总结最好保留在这里。
    if history_with_defense and 'server_with_defense' in locals():
        print("\n--- 防御效果总结 ---")
        final_reputations = server_with_defense.reputation_scores
        suspicious_clients_sorted = sorted(
            [(cid, rep) for cid, rep in final_reputations.items() if rep > 0],
            key=lambda item: item[1],
            reverse=True
        )
        if not suspicious_clients_sorted:
            print("    没有任何客户端的最终声誉分 > 0。")
        else:
            for cid, rep in suspicious_clients_sorted:
                is_malicious_str = "恶意" if cid < config["MALICIOUS_CLIENTS"] else "良性"
                print(f"    客户端 {cid} ({is_malicious_str}): 声誉分 = {rep}")


if __name__ == "__main__":
    run_experiment()
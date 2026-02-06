import json
import os
import sys
from pathlib import Path
import numpy as np

# ================= 配置 =================
# 你的数据集路径
DATASET_DIR = "/home/dx/waylen/SARM/transformed_datasets/pick_cube_sim_dense"
# 输出文件名
OUTPUT_FILENAME = "sarm_stats.json"

# 确保能导入本地 lerobot
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from lerobot.common.datasets.compute_stats import aggregate_stats
    print("成功导入本地 aggregate_stats")
except ImportError:
    print("Error: 无法导入 lerobot。请确保此脚本在 opensarm 目录下运行。")
    sys.exit(1)
# =======================================

def generate_stats():
    stats_path = Path(DATASET_DIR) / "meta/episodes_stats.jsonl"
    output_path = Path(DATASET_DIR) / "meta" / OUTPUT_FILENAME

    if not stats_path.exists():
        print(f"Error: 找不到源文件 {stats_path}")
        return

    print(f"Reading {stats_path}...")
    ep_stats_list = []
    
    with open(stats_path, "r") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)

                if "stats" not in raw:
                    continue

                
                episode_stats = {}

                for feature_key, feature_stats in raw["stats"].items():
                    converted = {}
                    for stat_key, value in feature_stats.items():
                        # 跳过非统计字段（如果你愿意也可以保留 count）
                        if stat_key not in ["min", "max", "mean", "std", "count"]:
                            continue

                        converted[stat_key] = np.asarray(value)

                    episode_stats[feature_key] = converted

                ep_stats_list.append(episode_stats)

    if not ep_stats_list:
        print("Error: 统计文件为空")
        return

    print(f"Aggregating stats from {len(ep_stats_list)} episodes...")
    # 计算全局统计 (Mean, Std, Min, Max)
    global_stats = aggregate_stats(ep_stats_list)

    # === 转换为 SARM 格式 ===
    sarm_stats = {
        "norm_stats": {}
    }

    # 映射关系: LeRobot Key -> SARM Key
    key_mapping = {
        "state": "state",
        "action": "actions"  # 注意这里有 's'
    }

    for src_key, target_key in key_mapping.items():
        if src_key in global_stats:
            src_data = global_stats[src_key]
            
            target_data = {
                "mean": src_data["mean"].tolist() if hasattr(src_data["mean"], "tolist") else src_data["mean"],
                "std": src_data["std"].tolist() if hasattr(src_data["std"], "tolist") else src_data["std"],
                # SARM 需要 q01/q99，我们用 min/max 替代
                "q01": src_data["min"].tolist() if hasattr(src_data["min"], "tolist") else src_data["min"],
                "q99": src_data["max"].tolist() if hasattr(src_data["max"], "tolist") else src_data["max"],
            }
            sarm_stats["norm_stats"][target_key] = target_data
            print(f"Processed: {src_key} -> {target_key}")
        else:
            print(f"Warning: 统计数据中缺少 {src_key}")

    # 保存文件
    with open(output_path, "w") as f:
        json.dump(sarm_stats, f, indent=2)

    print("\n========== DONE ==========")
    print(f"已生成 SARM 统计文件: {output_path}")
    print(f"请修改配置文件: state_norm_path: \"{output_path}\"")

if __name__ == "__main__":
    generate_stats()
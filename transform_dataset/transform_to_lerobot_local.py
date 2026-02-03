import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import os
import cv2

def resize_img(img, size=(256, 256)):
    # img: HWC numpy
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# ================= 核心修改：确保使用本地 lerobot =================
# 将当前目录加入系统路径，确保导入的是文件夹里的 lerobot，而不是 pip 安装的
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    print("成功导入本地 LeRobotDataset")
except ImportError as e:
    print("导入失败，请确保你在 opensarm 目录下运行此脚本，且该目录下有 lerobot 文件夹")
    raise e
# ==============================================================

# ================= 配置区域 =================
# 输入数据路径
INPUT_PKL_PATH = "/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/demo_data/flexiv_assembly_24_demos_2026-01-30_15-11-53.pkl"
# 输出路径 (SARM 训练配置中的 repo_id)
OUTPUT_DATASET_ROOT = "/home/dx/waylen/SARM/transformed_datasets/insert_wire_sparse"
DATASET_NAME = "insert_wire_sparse" # 本地数据集名称

# 任务描述
TASK_DESCRIPTION = "Insert the network cable into the motherboard."

# 键值映射: pkl_key -> lerobot_key
KEY_MAPPING = {
    "wrist_1": "observation.images.primary",
    "wrist_2": "observation.images.wrist",
    "state": "observation.state",
}

FPS = 10
# ===========================================
def flatten_state(state_dict: dict) -> np.ndarray:
    """
    After Quat2EulerWrapper:
    tcp_pose: (6,)
    gripper_pose: (1,) or (2,)
    """
    tcp_pose = np.asarray(state_dict[:6], dtype=np.float32).reshape(-1)
    gripper_pose = np.asarray([state_dict[12]], dtype=np.float32).reshape(-1)

    state = np.concatenate([tcp_pose, gripper_pose], axis=0)
    return state


def load_and_group_episodes(pkl_path):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    episodes = []
    current_episode = []

    for trans in tqdm(transitions, desc="Grouping episodes"):
        current_episode.append(trans)
        is_done = trans["dones"]
        if isinstance(is_done, np.ndarray):
            is_done = bool(is_done.item())
        if is_done:
            episodes.append(current_episode)
            current_episode = []

    print(f"Found {len(episodes)} complete episodes.")
    return episodes


def infer_features(first_step):
    obs = first_step["observations"]
    # print(obs["state"])
    action = first_step["actions"]

    # 定义基础特征
    # 注意：SARM/LeRobot 标准中，标量通常 shape 为 () 而不是 (1,)
    flat_state = flatten_state(obs["state"][-1])
    features = {
        "action": {
            "dtype": "float32", 
            "shape": tuple(action.shape), 
            "names": None
        },
        "done": {
            "dtype": "bool",   # 标准格式使用 bool
            "shape": (1,),      # 或者 ()
            "names": None
        },
        "reward": {
            "dtype": "float32", 
            "shape": (1,),      # SARM 进度值
            "names": None
        },
        "state": {
            "dtype": "float32", 
            "shape": tuple(flat_state.shape),     
            "names": None
        },
    }

    # 自动推断观察空间特征
    for src_key, target_key in KEY_MAPPING.items():
        if src_key not in obs:
            print(f"Warning: missing key {src_key}")
            continue

        sample = resize_img(obs[src_key][-1]) # 取最后一帧做样本

        if "images" in target_key:
            # 处理图像维度
            if sample.shape[-1] == 3:  # HWC -> (3, H, W)
                h, w, c = sample.shape
            else:  # CHW
                c, h, w = sample.shape

            features[target_key] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channel", "height", "width"],
            }
        # else:
        #     # 处理状态向量
        #     features[target_key] = {
        #         "dtype": "float32",
        #         "shape": sample.shape,
        #         "names": None,
        #     }

    return features


def convert_to_lerobot(episodes):
    # 1. 准备目录
    # Path(OUTPUT_DATASET_ROOT).mkdir(parents=True, exist_ok=True)
    
    # 2. 推断特征
    first_step = episodes[0][0]
    features = infer_features(first_step)

    # 3. 创建数据集实例 (使用本地 SARM 代码)
    print(f"Creating LeRobotDataset at {OUTPUT_DATASET_ROOT}...")
    dataset = LeRobotDataset.create(
        repo_id=DATASET_NAME,
        root=OUTPUT_DATASET_ROOT,
        fps=FPS,
        features=features,
        use_videos=True,
    )

    total_frames = 0

    # 4. 遍历并写入数据
    for ep_idx, episode in enumerate(tqdm(episodes, desc="Converting")):
        ep_len = len(episode)

        for frame_idx, step in enumerate(episode):
            # === 计算 SARM 进度 (Dense Reward) ===
            # 如果 ep_len=1, progress=1.0
            if ep_len > 1:
                progress = frame_idx / (ep_len - 1)
            else:
                progress = 1.0
            progress = float(np.clip(progress, 0.0, 1.0))
            # ====================================

            # 获取原始数据
            val_primary = resize_img(step["observations"]["wrist_1"][-1]) # HWC numpy
            val_wrist = resize_img(step["observations"]["wrist_2"][-1])   # HWC numpy
            # val_state = step["observations"]["state"][-1]
            raw_state = step["observations"]["state"][-1]
            val_state = flatten_state(raw_state)

            frame_data = {
                "action": torch.from_numpy(step["actions"]).float(),
                "done": torch.tensor([bool(step["dones"])]),
                "reward": torch.tensor([progress], dtype=torch.float32),

                "observation.images.primary": torch.from_numpy(val_primary).permute(2, 0, 1),
                "observation.images.wrist": torch.from_numpy(val_wrist).permute(2, 0, 1),
                # "observation.state": torch.from_numpy(val_state).float(),
                "state": torch.from_numpy(val_state).float(),

                "task": TASK_DESCRIPTION,
            }

            dataset.add_frame(frame_data)
            total_frames += 1

        # 一集结束，保存
        dataset.save_episode()

    # 5. 结束工作
    # SARM 版代码中没有 finalize()，但如果有 image_writer 需要停止
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    
    # 可以在这里显式调用 encode_videos，虽然 save_episode 内部通常会处理
    # dataset.encode_videos() 

    print("\n========== DONE ==========")
    print(f"Dataset saved to: {Path(OUTPUT_DATASET_ROOT)}")
    print(f"Total Episodes: {len(episodes)}")
    print(f"Total Frames: {total_frames}")
    
    # 验证文件是否生成
    if (Path(OUTPUT_DATASET_ROOT) / "meta/info.json").exists():
        print("Success: Meta info generated.")
    else:
        print("Warning: Meta info not found.")


# ================= 主入口 =================

if __name__ == "__main__":
    if not Path(INPUT_PKL_PATH).exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PKL_PATH}")

    episodes = load_and_group_episodes(INPUT_PKL_PATH)
    convert_to_lerobot(episodes)
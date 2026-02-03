import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ================= 配置区域 =================
INPUT_PKL_PATH = "/home/dx/waylen/conrft/examples/experiments/flexiv_assembly/demo_data/flexiv_assembly_24_demos_2026-01-30_15-11-53.pkl"
OUTPUT_DATASET_ROOT = "/home/dx/waylen/SARM/transformed_datasets/insert_wire_sparse"
DATASET_NAME = "flexiv_assembly_sarm_dataset"

TASK_DESCRIPTION = "Insert the network cable into the motherboard."

KEY_MAPPING = {
    "wrist_1": "observation.images.primary",
    "wrist_2": "observation.images.wrist",
    "state": "observation.state",
}

FPS = 10


# ================= 工具函数 =================

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
    action = first_step["actions"]

    features = {
        "action": {"dtype": "float32", "shape": tuple(action.shape), "names": None},
        "next.done": {"dtype": "int64", "shape": (1,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.images.primary": {"dtype": "video", "shape": (3, 256, 256), "names": None},
        "observation.images.wrist": {"dtype": "video", "shape": (3, 128, 128), "names": None},
        "observation.state": {"dtype": "float32", "shape": (19,), "names": None},
        # "task": {"dtype": "string", "shape": (1,), "names": None},
    }



    for src_key, target_key in KEY_MAPPING.items():
        if src_key not in obs:
            print(f"Warning: missing key {src_key}")
            continue

        sample = obs[src_key][-1]

        if "images" in target_key:
            print(sample.shape,sample.shape[-1])
            if sample.shape[-1] == 3:  # HWC
                h, w, c = sample.shape
            else:  # CHW
                c, h, w = sample.shape

            features[target_key] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channel", "height", "width"],
            }
        else:
            features[target_key] = {
                "dtype": "float32",
                "shape": sample.shape,
                "names": None,
            }

    return features


def convert_to_lerobot(episodes):
    first_step = episodes[0][0]
    features = infer_features(first_step)

    dataset = LeRobotDataset.create(
        repo_id=DATASET_NAME,
        root=OUTPUT_DATASET_ROOT,
        fps=FPS,
        features=features,
        # task=TASK_DESCRIPTION,
        use_videos=True,
    )

    total_frames = 0

    for ep_idx, episode in enumerate(tqdm(episodes, desc="Converting")):
        ep_len = len(episode)

        for frame_idx, step in enumerate(episode):
            if ep_len > 1:
                progress = frame_idx / (ep_len - 1)
            else:
                progress = 1.0

            progress = float(np.clip(progress, 0.0, 1.0))
            val_primary = step["observations"]["wrist_1"][-1]
            val_wrist = step["observations"]["wrist_2"][-1]
            val_state = step["observations"]["state"][-1]

            frame_data = {
                "action": torch.from_numpy(step["actions"]).float(),

                "next.done": np.array([int(step["dones"])], dtype=np.int64),
                "next.reward": np.array([progress], dtype=np.float32),

                "observation.images.primary": torch.from_numpy(val_primary).permute(2, 0, 1),
                "observation.images.wrist": torch.from_numpy(val_wrist).permute(2, 0, 1),
                "observation.state": torch.from_numpy(val_state).float(),

                "task": TASK_DESCRIPTION
            }

            dataset.add_frame(frame_data)
            total_frames += 1

        dataset.save_episode()

    dataset.finalize()

    print("\n========== DONE ==========")
    print(f"Dataset path: {Path(OUTPUT_DATASET_ROOT) / DATASET_NAME}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")


# ================= 主入口 =================

if __name__ == "__main__":
    if not Path(INPUT_PKL_PATH).exists():
        raise FileNotFoundError(INPUT_PKL_PATH)

    episodes = load_and_group_episodes(INPUT_PKL_PATH)
    convert_to_lerobot(episodes)

import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import os
import cv2
import time

def resize_img(img, size=(256, 256)):
    # img: HWC numpy
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# ================= 核心修改：确保使用本地 lerobot =================
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
INPUT_PKL_PATH = "/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2026-02-05_13-55-18.pkl"
OUTPUT_DATASET_ROOT = "/home/dx/waylen/SARM/transformed_datasets/pick_cube_sim_dense"
DATASET_NAME = "pick_cube_sim"

TASK_DESCRIPTION = "Pick up the cube and lift it."

KEY_MAPPING = {
    "wrist_1": "observation.images.primary",
    "wrist_2": "observation.images.wrist",
    "state": "observation.state",
}

FPS = 10
NUM_STAGES = 3  # 【设置】总阶段数。例如2阶段，意味着中间需要按1次空格进行分割
# ===========================================

def flatten_state(state_dict: dict) -> np.ndarray:
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
    action = first_step["actions"]
    flat_state = flatten_state(obs["state"][-1])
    
    features = {
        "action": {"dtype": "float32", "shape": tuple(action.shape), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "state": {"dtype": "float32", "shape": tuple(flat_state.shape), "names": None},
    }

    for src_key, target_key in KEY_MAPPING.items():
        if src_key not in obs: continue
        sample = resize_img(obs[src_key][-1])
        if "images" in target_key:
            if sample.shape[-1] == 3: h, w, c = sample.shape
            else: c, h, w = sample.shape
            features[target_key] = {"dtype": "video", "shape": (c, h, w), "names": ["channel", "height", "width"]}
            
    return features

# ================= 新增：交互式阶段标注函数 =================
def get_episode_split_points(episode, ep_idx, total_eps):
    """
    可视化播放一集，允许用户按空格标记分割点。
    返回一个包含分割帧索引的列表（包括0和结尾）。
    例如 NUM_STAGES=2, 长度100 -> 返回 [0, 45, 100]
    """
    needed_splits = NUM_STAGES - 1
    window_name = f"Annotator: Ep {ep_idx+1}/{total_eps} - Need {needed_splits} Splits"
    
    while True:
        split_indices = []
        frames_list = []
        
        # 预加载图像以便播放流畅
        for step in episode:
            # 获取 primary image 用于显示 (注意 convert BGR for OpenCV display)
            img = step["observations"]["wrist_1"][-1]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames_list.append(img_bgr)
            
        print(f"\n>>> 开始标注第 {ep_idx+1} 集。请按【空格键】标记 {needed_splits} 个分割点。按 'r' 重来。")
        
        paused = False
        idx = 0
        while idx < len(frames_list):
            frame = frames_list[idx].copy()
            
            # 在图像上绘制 UI 信息
            info_text = f"Frame: {idx}/{len(frames_list)} | Splits: {len(split_indices)}/{needed_splits}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Space: Split | R: Retry", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            time.sleep(0.01)
            
            cv2.imshow(window_name, frame)
            
            # 等待按键 (调节播放速度，这里 30ms 约为 33fps，可调大变慢)
            wait_ms = 0 if paused else 40 
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == ord(' '):  # Spacebar
                if len(split_indices) < needed_splits:
                    split_indices.append(idx)
                    print(f"  -> Split marked at frame {idx}")
                    # 视觉反馈
                    cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 20, (0, 0, 255), -1)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(200) 
                else:
                    print("  Warning: 已达到最大分割点数量，忽略此次按键。")
            
            elif key == ord('r'): # Retry
                print("  [重置] 本集重新开始...")
                split_indices = []
                idx = 0
                continue
                
            elif key == ord('p'): # Pause
                paused = not paused
                
            elif key == 27: # ESC to force quit script
                cv2.destroyAllWindows()
                sys.exit("标注被用户终止。")

            if not paused:
                idx += 1

        # 本集结束，验证分割点数量
        if len(split_indices) == needed_splits:
            print(f"第 {ep_idx+1} 集标注完成。分割点: {split_indices}")
            # 构建完整的边界列表 [0, split1, split2, ..., length]
            boundaries = [0] + split_indices + [len(episode)]
            boundaries.sort() # 确保有序
            cv2.destroyWindow(window_name)
            return boundaries
        else:
            print(f"错误: 需要 {needed_splits} 个分割点，但只标记了 {len(split_indices)} 个。请重新标注本集！")
            cv2.waitKey(1000) # 暂停一秒让用户看到错误信息
            # 循环会自动回到 while True 开头重试

def calculate_multi_stage_reward(frame_idx, boundaries):
    """
    根据当前帧和边界计算累积 Reward。
    boundaries: [0, split_1, split_2, ..., total_len]
    如果 frame_idx 在 stage 0 (0 -> split_1): reward = 0 + progress(0~1)
    如果 frame_idx 在 stage 1 (split_1 -> split_2): reward = 1 + progress(0~1)
    """
    total_stages = len(boundaries) - 1
    
    # 找到当前帧所属的阶段
    current_stage = 0
    for i in range(total_stages):
        start = boundaries[i]
        end = boundaries[i+1]
        if start <= frame_idx < end:
            current_stage = i
            break
        # 处理最后一帧的情况 (通常是 done 帧)
        if frame_idx == end and i == total_stages - 1:
            current_stage = i
            
    # 获取当前阶段的起止点
    stage_start = boundaries[current_stage]
    stage_end = boundaries[current_stage + 1]
    
    # 计算阶段内的局部进度 [0, 1]
    stage_len = stage_end - stage_start
    if stage_len > 0:
        local_progress = (frame_idx - stage_start) / stage_len
    else:
        local_progress = 1.0 # 防止除以0
        
    local_progress = np.clip(local_progress, 0.0, 1.0)
    
    # 最终 Reward = 阶段索引 + 局部进度
    # Stage 0: 0.0 ~ 1.0
    # Stage 1: 1.0 ~ 2.0
    cumulative_reward = float(current_stage + local_progress)
    
    return cumulative_reward

# ==============================================================

def convert_to_lerobot(episodes):
    # 1. 初始化
    first_step = episodes[0][0]
    features = infer_features(first_step)

    print(f"Creating LeRobotDataset at {OUTPUT_DATASET_ROOT}...")
    dataset = LeRobotDataset.create(
        repo_id=DATASET_NAME,
        root=OUTPUT_DATASET_ROOT,
        fps=FPS,
        features=features,
        use_videos=True,
    )

    total_frames = 0
    total_eps = len(episodes)

    # 2. 遍历并处理
    for ep_idx, episode in enumerate(episodes):
        
        # === 核心修改：先进行人工标注获取分割点 ===
        # boundaries 格式如 [0, 45, 100]
        boundaries = get_episode_split_points(episode, ep_idx, total_eps)
        # ========================================

        for frame_idx, step in enumerate(episode):
            # === 核心修改：计算多阶段累积 Reward ===
            reward_val = calculate_multi_stage_reward(frame_idx, boundaries)
            # ====================================

            # 获取原始数据
            val_primary = resize_img(step["observations"]["wrist_1"][-1]) 
            val_wrist = resize_img(step["observations"]["wrist_2"][-1])   
            raw_state = step["observations"]["state"][-1]
            val_state = flatten_state(raw_state)

            frame_data = {
                "action": torch.from_numpy(step["actions"]).float(),
                "done": torch.tensor([bool(step["dones"])]),
                "reward": torch.tensor([reward_val], dtype=torch.float32), # 写入新的 reward

                "observation.images.primary": torch.from_numpy(val_primary).permute(2, 0, 1),
                "observation.images.wrist": torch.from_numpy(val_wrist).permute(2, 0, 1),
                "state": torch.from_numpy(val_state).float(),
                "task": TASK_DESCRIPTION,
            }

            dataset.add_frame(frame_data)
            total_frames += 1

        dataset.save_episode()
    
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
        
    # 关闭 OpenCV 窗口
    cv2.destroyAllWindows()

    print("\n========== DONE ==========")
    print(f"Dataset saved to: {Path(OUTPUT_DATASET_ROOT)}")
    print(f"Total Episodes: {len(episodes)}")
    print(f"Total Frames: {total_frames}")
    
    if (Path(OUTPUT_DATASET_ROOT) / "meta/info.json").exists():
        print("Success: Meta info generated.")
    else:
        print("Warning: Meta info not found.")


# ================= 主入口 =================
if __name__ == "__main__":
    if not Path(INPUT_PKL_PATH).exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PKL_PATH}")

    # 检查是否有 GUI 环境 (cv2.imshow 需要)
    if os.environ.get('DISPLAY', '') == '':
        print('Warning: No display environment detected. cv2.imshow might fail.')

    episodes = load_and_group_episodes(INPUT_PKL_PATH)
    convert_to_lerobot(episodes)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pickle
import time
import os
from tqdm import tqdm
from stage_srv_cpp.srv import StageInference
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


TARGET_RES = (256, 256) 
KEY_MAPPING = {
    "wrist_1": "observation.images.primary",
    "wrist_2": "observation.images.wrist",
    "state": "observation.state",
}

class StageClientNode(Node):
    def __init__(self):
        super().__init__("stage_client_node")
        self.cli = self.create_client(StageInference, "stage_inference")
        self.bridge = CvBridge()
        
        # 等待服务上线
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.get_logger().info('StageInference service is available.')

    def send_request(self, primary_img, wrist_img, state_vector):
        """
        构造并发送请求
        """
        req = StageInference.Request()
        req.image_primary = self.bridge.cv2_to_imgmsg(primary_img, encoding="rgb8")
        req.image_wrist = self.bridge.cv2_to_imgmsg(wrist_img, encoding="rgb8")
        req.state = state_vector.tolist()
        self.future = self.cli.call_async(req)
        
        return self.future

def resize_img(img):
    """
    调整图像大小以适配模型输入
    """
    if img is None:
        return np.zeros((TARGET_RES[1], TARGET_RES[0], 3), dtype=np.uint8)
    resized = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_LINEAR)
    return resized

def flatten_state(state_data) -> np.ndarray:
    """
    根据你提供的逻辑提取状态
    注意：这里假设 state_data 是 numpy array 或 list
    """
    s = np.array(state_data)
    tcp_pose = np.asarray(s[:6], dtype=np.float32).reshape(-1)

    if len(s) > 12:
        gripper_pose = np.asarray([s[12]], dtype=np.float32).reshape(-1)
    else:
        gripper_pose = np.array([0.0], dtype=np.float32)
        
    state = np.concatenate([tcp_pose, gripper_pose], axis=0)
    return state

def load_and_group_episodes(pkl_path):
    """
    你的原始加载函数
    """
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
        if isinstance(is_done, (bool, np.bool_)) and is_done:
            episodes.append(current_episode)
            current_episode = []
        elif isinstance(is_done, (int, np.integer)) and is_done == 1:
            episodes.append(current_episode)
            current_episode = []

    if len(current_episode) > 0:
        episodes.append(current_episode)

    print(f"Found {len(episodes)} complete episodes.")
    return episodes


def main(args=None):
    rclpy.init(args=args)
    pkl_file_path = "/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2026-02-05_13-55-18.pkl"  # <--- 请修改这里
    
    if not os.path.exists(pkl_file_path):
        print(f"Error: File {pkl_file_path} not found.")
        return
    episodes = load_and_group_episodes(pkl_file_path)
    
    # 3. 初始化 ROS 节点
    client_node = StageClientNode()
    
    try:
        for ep_idx, episode in enumerate(episodes):
            print(f"Processing Episode {ep_idx + 1}/{len(episodes)}")
            for step_idx, step_data in enumerate(episode):
                obs = step_data["observations"]
                
                pkl_key_primary = [k for k, v in KEY_MAPPING.items() if k == "wrist_1"][0]
                raw_primary = obs.get(pkl_key_primary, None)
                if isinstance(raw_primary, (list, np.ndarray)) and len(np.shape(raw_primary)) == 4:
                        raw_primary = raw_primary[-1]
                img_primary_processed = resize_img(raw_primary)

                pkl_key_wrist = [k for k, v in KEY_MAPPING.items() if k == "wrist_2"][0]
                raw_wrist = obs.get(pkl_key_wrist, None)
                if isinstance(raw_wrist, (list, np.ndarray)) and len(np.shape(raw_wrist)) == 4:
                        raw_wrist = raw_wrist[-1]
                img_wrist_processed = resize_img(raw_wrist)

                raw_state = obs.get("state", [])
                # print(obs.keys())
                if isinstance(raw_state, (list, np.ndarray)) and len(np.shape(raw_state)) > 1:
                        raw_state = raw_state[-1]
                
                state_vec = flatten_state(raw_state)
                # print(img_primary_processed.shape, img_wrist_processed.shape, state_vec)

                future = client_node.send_request(
                    primary_img=img_primary_processed,
                    wrist_img=img_wrist_processed,
                    state_vector=state_vec
                )

                rclpy.spin_until_future_complete(client_node, future)
                # time.sleep(0.1)  # 可选：稍作延时，避免过快发送请求

                if future.result() is not None:
                    response = future.result()
                    print(f"  Step {step_idx}: Stage Inference Result -> {response.stage}")
                else:
                    client_node.get_logger().error(f"Service call failed at episode {ep_idx} step {step_idx}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
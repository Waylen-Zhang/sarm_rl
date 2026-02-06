import gymnasium as gym  # 或者 import gym
from gym import Env
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import collections
from stage_srv_cpp.srv import StageInference
import uuid
import os

# 引入你的 Service 类型
from stage_srv_cpp.srv import StageInference

TARGET_RES = (256, 256)
KEY_MAPPING = {
    "wrist_1": "observation.images.primary",
    "wrist_2": "observation.images.wrist",
    "state": "observation.state",
}

class StageAwareRewardWrapper(gym.Wrapper):
    """
    此 Wrapper 结合了基于图像的最终奖励分类器和基于 ROS Service 的过程阶段奖励。
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz=None, 
                 stage_buffer_size=5, node_name="gym_stage_client"):
        super().__init__(env)
        
        # 1. 原始参数
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

        # 2. ROS2 初始化
        if not rclpy.ok():
            rclpy.init()
        
        # 创建一个专属的节点用于通信，避免与外部节点冲突
        self.node = rclpy.create_node(node_name)
        self.cli = self.node.create_client(StageInference, "stage_inference")
        self.bridge = CvBridge()

        # 等待服务上线 (可以在 reset 中再次检查，这里为了初始化先检查一次)
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().warn('StageInference service not immediately available.')

        # 3. 阶段追踪逻辑参数
        self.stage_buffer_size = stage_buffer_size
        self.stage_buffer = collections.deque(maxlen=stage_buffer_size)
        
        # 状态追踪变量
        self.max_reached_stage = 0
        self.current_stable_stage = 0

    # --- 数据处理辅助函数 (复用你提供的代码) ---
    @staticmethod
    def _resize_img(img):
        if img is None:
            return np.zeros((TARGET_RES[1], TARGET_RES[0], 3), dtype=np.uint8)
        resized = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def _flatten_state(state_data) -> np.ndarray:
        s = np.array(state_data)
        # 处理可能存在的 batch 维度或历史维度
        if s.ndim > 1:
            s = s.flatten()
            
        # 安全截取，防止索引越界
        if len(s) >= 6:
            tcp_pose = np.asarray(s[:6], dtype=np.float32).reshape(-1)
        else:
            tcp_pose = np.zeros(6, dtype=np.float32)

        if len(s) > 12:
            gripper_pose = np.asarray([s[12]], dtype=np.float32).reshape(-1)
        else:
            gripper_pose = np.array([0.0], dtype=np.float32)
            
        state = np.concatenate([tcp_pose, gripper_pose], axis=0)
        return state

    def _get_obs_inputs(self, obs):
        """从 Gym 的 observation dict 中提取图像和状态"""

        raw_primary = obs.get("observation.images.primary", None)
        if raw_primary is None and "images" in obs: 
             raw_primary = obs["images"].get("primary", None)

        if isinstance(raw_primary, (list, np.ndarray)) and len(np.shape(raw_primary)) == 4:
            raw_primary = raw_primary[-1]
        img_primary_processed = self._resize_img(raw_primary)

        raw_wrist = obs.get("observation.images.wrist", None)
        if raw_wrist is None and "images" in obs:
            raw_wrist = obs["images"].get("wrist", None)

        if isinstance(raw_wrist, (list, np.ndarray)) and len(np.shape(raw_wrist)) == 4:
            raw_wrist = raw_wrist[-1]
        img_wrist_processed = self._resize_img(raw_wrist)

        raw_state = obs.get("observation.state", [])
        if raw_state is None and "state" in obs:
            raw_state = obs["state"]
            
        if isinstance(raw_state, (list, np.ndarray)) and len(np.shape(raw_state)) > 1:
            raw_state = raw_state[-1]
        state_vec = self._flatten_state(raw_state)

        return img_primary_processed, img_wrist_processed, state_vec

    def _call_stage_service(self, primary_img, wrist_img, state_vec):
        """调用 ROS Service 并阻塞等待结果"""
        req = StageInference.Request()
        req.image_primary = self.bridge.cv2_to_imgmsg(primary_img, encoding="rgb8")
        req.image_wrist = self.bridge.cv2_to_imgmsg(wrist_img, encoding="rgb8")
        req.state = state_vec.tolist()

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            return future.result().stage
        else:
            self.node.get_logger().error("Stage service call failed")
            return -1

    def compute_final_reward(self, obs):
        """原始的最终奖励逻辑"""
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        
        # 1. 环境 Step
        obs, rew, done, truncated, info = self.env.step(action)

        # 2. 计算过程奖励 (Stage Reward)
        stage_reward = 0.0
        current_pred_stage = -1
        
        try:
            # 提取数据
            p_img, w_img, s_vec = self._get_obs_inputs(obs)
            # 调用服务
            current_pred_stage = self._call_stage_service(p_img, w_img, s_vec)
        except Exception as e:
            self.node.get_logger().error(f"Error in stage inference: {e}")

        # 更新 Buffer
        if current_pred_stage != -1:
            self.stage_buffer.append(current_pred_stage)

        # 判断是否稳定：Buffer 满且所有元素相同
        is_stable = (len(self.stage_buffer) == self.stage_buffer_size) and \
                    (len(set(self.stage_buffer)) == 1)

        if is_stable:
            stable_stage = self.stage_buffer[0]
            
            # 更新当前稳定阶段（如果不考虑向上跳变，这里可能需要额外逻辑，
            # 但通常 Buffer 已经起到了滤波作用。这里假设只要稳定就更新）
            self.current_stable_stage = stable_stage

            # 触发奖励逻辑：只有当稳定阶段 超过 历史最大阶段时
            if self.current_stable_stage > self.max_reached_stage:
                stage_reward = 0.2
                self.max_reached_stage = self.current_stable_stage
                # 可选：打印日志
                print(f"Stage Reached: {self.max_reached_stage}, Reward: +0.2")

        # 3. 计算最终奖励 (Final Reward)
        final_reward = self.compute_final_reward(obs)
        
        # 4. 汇总
        # 注意：done 的逻辑只由 final_reward 决定，与过程奖励无关
        done = done or bool(final_reward)
        info['succeed'] = bool(final_reward)
        
        # 记录阶段信息到 info 中方便调试
        info['stage_reward'] = stage_reward
        info['current_stage'] = self.current_stable_stage
        info['max_stage'] = self.max_reached_stage
        
        total_reward = final_reward + stage_reward

        # 5. 频率控制
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self.max_reached_stage = 0
        self.current_stable_stage = 0
        self.stage_buffer.clear()
        
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info

    def close(self):
        self.node.destroy_node()
        super().close()




class SimStageAwareRewardWrapper(gym.Wrapper):
    """
    此 Wrapper 专注于通过 ROS2 Service 获取阶段预测，并计算过程奖励。
    它不处理最终任务成功的判断（假设底层环境已经处理了）。
    """

    def __init__(self, env: Env, target_hz=None, stage_buffer_size=5, node_name_prefix="gym_stage_client"):
        super().__init__(env)
        self.target_hz = target_hz

        if not rclpy.ok():
            rclpy.init()
        
        unique_node_name = f"{node_name_prefix}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        self.node = rclpy.create_node(unique_node_name)
        
        self.cli = self.node.create_client(StageInference, "stage_inference")
        self.bridge = CvBridge()

        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().warn(f'[{unique_node_name}] StageInference service not ready yet.')
        else:
            self.node.get_logger().info(f'[{unique_node_name}] Connected to StageInference service.')

        self.stage_buffer_size = stage_buffer_size
        self.stage_buffer = collections.deque(maxlen=stage_buffer_size)
        
        self.max_reached_stage = -1
        self.current_stable_stage = -1

    # ================= 数据处理 =================
    @staticmethod
    def _resize_img(img):
        if img is None:
            return np.zeros((TARGET_RES[1], TARGET_RES[0], 3), dtype=np.uint8)
        resized = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def _flatten_state(state_data) -> np.ndarray:
        s = np.array(state_data)
        if s.ndim > 1: s = s.flatten() # 展平
            
        # 提取 TCP pose (前6维)
        if len(s) >= 6:
            tcp_pose = np.asarray(s[:6], dtype=np.float32).reshape(-1)
        else:
            tcp_pose = np.zeros(6, dtype=np.float32)

        if len(s) > 12:
            gripper_pose = np.asarray([s[12]], dtype=np.float32).reshape(-1)
        else:
            # 仿真中如果没有这一维，给默认值
            gripper_pose = np.array([0.0], dtype=np.float32)
            
        state = np.concatenate([tcp_pose, gripper_pose], axis=0)
        return state

    def _get_obs_inputs(self, obs):
        """兼容性地提取图像和状态"""
        # 1. Primary Image
        raw_primary = obs.get("observation.images.primary", None)
        if raw_primary is None and "images" in obs: raw_primary = obs["images"].get("primary", None)
        if isinstance(raw_primary, (list, np.ndarray)) and len(np.shape(raw_primary)) == 4:
            raw_primary = raw_primary[-1]
        
        # 2. Wrist Image
        raw_wrist = obs.get("observation.images.wrist", None)
        if raw_wrist is None and "images" in obs: raw_wrist = obs["images"].get("wrist", None)
        if isinstance(raw_wrist, (list, np.ndarray)) and len(np.shape(raw_wrist)) == 4:
            raw_wrist = raw_wrist[-1]

        # 3. State
        raw_state = obs.get("observation.state", None)
        if raw_state is None: raw_state = obs.get("state", [])
        if isinstance(raw_state, (list, np.ndarray)) and len(np.shape(raw_state)) > 1:
            raw_state = raw_state[-1]

        return (self._resize_img(raw_primary), 
                self._resize_img(raw_wrist), 
                self._flatten_state(raw_state))

    # ================= ROS 通信 =================
    def _call_stage_service(self, primary_img, wrist_img, state_vec):
        """构造请求并阻塞等待"""
        req = StageInference.Request()
        req.image_primary = self.bridge.cv2_to_imgmsg(primary_img, encoding="rgb8")
        req.image_wrist = self.bridge.cv2_to_imgmsg(wrist_img, encoding="rgb8")
        req.state = state_vec.tolist()

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            print("Stage inference result:", future.result().stage)
            return future.result().stage
        else:
            return -1

    # ================= Gym 接口 =================
    def step(self, action):
        step_start = time.time()
        
        # 1. 执行底层环境的 step
        # 这里假设 sim 环境已经根据位置计算了任务完成奖励 (rew) 和结束标志 (done)
        obs, rew, done, truncated, info = self.env.step(action)
        
        # 2. 计算过程奖励
        stage_reward = 0.0
        
        # 只有在未结束时才计算过程奖励（或者根据需求，即使 done 了最后一帧也算）
        # 这里加上 try-except 防止 ROS 通信挂掉影响主训练循环
        try:
            p_img, w_img, s_vec = self._get_obs_inputs(obs)
            current_stage_pred = self._call_stage_service(p_img, w_img, s_vec)
            
            if current_stage_pred != -1:
                self.stage_buffer.append(current_stage_pred)
                
            # 稳定性判断
            if (len(self.stage_buffer) == self.stage_buffer_size) and (len(set(self.stage_buffer)) == 1):
                stable_stage = self.stage_buffer[0]
                
                # 触发条件：当前稳定阶段 > 历史最大阶段
                if stable_stage > self.max_reached_stage:
                    print(f"Stage advanced from {self.max_reached_stage} to {stable_stage}")
                    stage_reward = 0.2
                    self.max_reached_stage = stable_stage
                    
        except Exception as e:
            self.node.get_logger().error(f"Stage inference error: {e}")

        # 3. 融合奖励
        total_reward = rew + stage_reward
        
        # 4. 更新 Info (用于调试和记录)
        info['stage_reward'] = stage_reward
        info['max_reached_stage'] = self.max_reached_stage
        
        # 注意：这里的 done 完全由底层环境决定，不被过程奖励影响
        # 如果底层环境没有设 success key，这里可以补充一下
        if 'succeed' not in info:
            info['succeed'] = done # 简单假设 done 即成功，或根据 rew > 0 判断

        # 5. 频率控制
        if self.target_hz is not None:
            elapsed = time.time() - step_start
            sleep_time = max(0, 1/self.target_hz - elapsed)
            time.sleep(sleep_time)

        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        # 重置内部状态
        self.max_reached_stage = -1
        self.current_stable_stage = -1
        self.stage_buffer.clear()
        
        return self.env.reset(**kwargs)

    def close(self):
        self.node.destroy_node()
        super().close()
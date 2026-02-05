import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from termcolor import cprint
from collections import deque

from tqdm import tqdm
# import wandb

from lerobot.common.datasets.rm_lerobot_dataset import FrameGapLeRobotDataset 
from utils.data_utils import get_valid_episodes, split_train_eval_episodes, adapt_lerobot_batch_sarm
from utils.train_utils import set_seed, save_ckpt, get_normalizer_from_calculated, plot_episode_result_raw_data, plot_episode_result
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data, normalize_sparse, normalize_dense
from models.subtask_estimator import SubtaskTransformer
from models.stage_estimator import StageTransformer
from models.clip_encoder import FrozenCLIPEncoder
from utils.make_demo_video import produce_video
from utils.pred_smoother import RegressionConfidenceSmoother
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from stage_srv_cpp.srv import StageInference

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"

def infinite_loader(dl):
    """Yield batches forever; reshuffles each pass if dl.shuffle=True."""
    while True:
        for b in dl:
            yield b

# ============================================================
# ROS Node (Service Server)
# ============================================================

class StageServiceNode(Node):
    def __init__(self, workspace):
        super().__init__("stage_service_node")

        self.ws = workspace
        self.bridge = CvBridge()

        self.srv = self.create_service(
            StageInference,
            "stage_inference",
            self._handle_request
        )

        self.get_logger().info("StageInference service ready.")
    @torch.no_grad()  
    def _handle_request(self, req, res):
        # ---- ROS Image -> Tensor ----
        img_primary = self.ws.rosimg_to_tensor(req.image_primary)
        img_wrist = self.ws.rosimg_to_tensor(req.image_wrist)

        # ---- state ----
        state = torch.tensor(
            req.state,
            dtype=torch.float32,
            device=self.ws.device
        ).unsqueeze(0)

        # ---- inference ----
        stage = self.ws.infer_stage(
            img_primary=img_primary,
            img_wrist=img_wrist,
            state=state
        )

        res.stage = stage
        return res


# ============================================================
# Workspace
# ============================================================

class SARMWorkspace:
    """
    Unified workspace:
      - eval()
      - eval_raw_data()
      - deploy()  <-- ROS Service mode
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(
            cfg.general.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Init] Using device: {self.device}")

        set_seed(cfg.general.seed)

        self.camera_names = cfg.general.camera_names

        # These will be initialized in deploy()
        self.clip_encoder = None
        self.stage_model = None
        self.state_normalizer = None
        self.node = None

        self.n_obs_steps = cfg.model.n_obs_steps

        # history buffers (for deploy)
        self.img_emb_buffer = deque(maxlen=self.n_obs_steps)
        self.state_buffer = deque(maxlen=self.n_obs_steps)

    # ========================================================
    # ROS deploy entry
    # ========================================================

    def deploy(self):
        """
        ROS deployment entry, called from deploy.py
        """
        print("[DEPLOY] Initializing ROS Stage Service...")

        self._init_models_for_deploy()

        # Create ROS node
        self.node = StageServiceNode(self)

        print("[DEPLOY] ROS Stage Service ready.")

    # ========================================================
    # Model initialization (extracted from eval logic)
    # ========================================================

    def _init_models_for_deploy(self):
        cfg = self.cfg
        
        # ---- state normalizer ----
        self.state_normalizer = get_normalizer_from_calculated(
            cfg.general.state_norm_path,
            self.device
        )
        
        # ---- CLIP encoder ----
        self.clip_encoder = FrozenCLIPEncoder(
            cfg.encoders.vision_ckpt,
            self.device
        )
        # ---- stage model ----
        self.stage_model = StageTransformer(
            d_model=cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=cfg.model.state_dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_cameras=len(cfg.general.camera_names),
            num_classes_sparse=cfg.model.num_classes_sparse,
            num_classes_dense=cfg.model.num_classes_dense,
        ).to(self.device)
        reward_model_path = Path(cfg.eval.ckpt_path)
        ckpt_name = cfg.eval.stage_model
        stage_model_path = reward_model_path / ckpt_name
        ckpt = torch.load(
            stage_model_path,
            map_location=self.device
        )
        self.stage_model.load_state_dict(ckpt["model"])
        self.stage_model.eval()

        print("[DEPLOY] Models loaded successfully.")

    # ========================================================
    # Single-step inference (核心逻辑)
    # ========================================================

    @torch.no_grad()
    def infer_stage(self, img_primary, img_wrist, state):
        """
        img_primary: (1, 3, H, W)
        img_wrist:   (1, 3, H, W)
        state:       (1, state_dim)
        """

        # ---- image encoding ----
        img_list = [img_primary, img_wrist]
        imgs_all = torch.cat(img_list, dim=0)     # (2, 3, H, W)

        img_emb = self.clip_encoder.encode_image(imgs_all)
        if hasattr(img_emb, "pooler_output"):
            img_emb = img_emb.pooler_output
        elif hasattr(img_emb, "last_hidden_state"):
            img_emb = img_emb.last_hidden_state[:, 0]
        # img_emb = img_emb.view(1, 2, 1, -1)       # (B=1, N=2, T=1, D)
        img_emb = img_emb.view(1, 2, -1)
        # self.img_emb_buffer.append(img_emb)     # each: (1, 2, D)
        self.img_emb_buffer.append(img_emb.detach()) 

        # ---- text encoding ----
        lang_emb = self.clip_encoder.encode_text(
            ["insert wire into motherboard"]
        )
        if hasattr(lang_emb, "pooler_output"):
            lang_emb = lang_emb.pooler_output
        elif hasattr(lang_emb, "last_hidden_state"):
            lang_emb = lang_emb.last_hidden_state[:, 0]

        # ---- state normalize ----
        state = self.state_normalizer.normalize(state)
        # print(img_emb.shape, lang_emb.shape, state.shape)
        self.state_buffer.append(state.detach())   # each: (1, state_dim)

        img_emb_seq = torch.stack(
            list(self.img_emb_buffer), dim=2
        ).detach()

        state_seq = torch.stack(
            list(self.state_buffer), dim=1
        ).detach()
        T = img_emb_seq.shape[2]
        lens = torch.tensor([T], device=self.device)

        # lens = torch.tensor([1], device=self.device)

        # ---- stage forward ----
        stage_prob = self.stage_model(
            img_emb_seq,
            lang_emb,
            state_seq,
            lens,
            scheme=self.cfg.eval.model_type
        ).softmax(dim=-1)

        stage_idx = stage_prob.argmax(dim=-1)

        return int(int(stage_idx[0, -1].item()))

    # ========================================================
    # Utils
    # ========================================================

    def rosimg_to_tensor(self, img_msg: Image):
        """
        sensor_msgs/Image -> torch.FloatTensor (1, 3, H, W)
        """
        cv_img = self.node.bridge.imgmsg_to_cv2(
            img_msg,
            desired_encoding="rgb8"
        )
        tensor = (
            torch.from_numpy(cv_img)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        return tensor.unsqueeze(0).to(self.device)

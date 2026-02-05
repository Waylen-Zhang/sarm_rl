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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"

def infinite_loader(dl):
    """Yield batches forever; reshuffles each pass if dl.shuffle=True."""
    while True:
        for b in dl:
            yield b

            
class SARMWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def gen_stage_emb(self, num_classes, trg):
        """
        Returns stage_onehot with a modality dim (B, 1, T, C).
        """
        # integer part of float targets -> [0, C-1]
        idx = trg.long().clamp(min=0, max=num_classes - 1)   # (B, T)

        C = num_classes
        # identity-lookup one-hot
        stage_onehot = torch.eye(C, device=trg.device)[idx]            # (B, T, C)
        stage_onehot = stage_onehot.unsqueeze(1)                       # (B, 1, T, C)
        return stage_onehot

    # Evaluate whole trajectory from demo data, generating video
    def eval(self):
        import random
        cfg = self.cfg
        model_type = cfg.eval.model_type
        dataset_type = cfg.eval.dataset_type
        if dataset_type == "sparse":
            repo_id = cfg.general.repo_id_sparse
        else:
            repo_id = cfg.general.repo_id_dense
        
        
        valid_episodes = get_valid_episodes(repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)
        dataset_val = FrameGapLeRobotDataset(repo_id=repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               annotation_list=cfg.model.sparse_annotation_list,
                                               task_name=cfg.general.task_name,
                                               video_eval=True,
                                               video_backend="pyav")
        
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512
        
        subtask_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.subtask_model
        stage_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.stage_model
        
            
        if model_type == "sparse":
            num_classes = cfg.model.num_classes_sparse
        else:
            num_classes = cfg.model.num_classes_dense

        # --- reward_model ---
        subtask_model = SubtaskTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  num_classes_sparse=cfg.model.num_classes_sparse,
                                  num_classes_dense=cfg.model.num_classes_dense
                                  ).to(self.device)

        # Load checkpoints
        subtask_ckpt = torch.load(subtask_model_path, map_location=self.device)
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        subtask_model.load_state_dict(subtask_ckpt["model"])
        stage_model.load_state_dict(stage_ckpt["model"])
        subtask_model.to(self.device)
        stage_model.to(self.device)
        subtask_model.eval(); stage_model.eval()

        # save path
        rollout_save_dir =  Path(self.save_dir) / "eval_video"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = subtask_model_path.stem; ckpt_note_path = rollout_save_dir / "ckpt_note.txt"
        with open(ckpt_note_path, "w", encoding="utf-8") as f:
            f.write(f"subtask model: {ckpt_name}\n")
            
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        evaled_list = []

        for i in range(cfg.eval.run_times):
            ep_index = random.choice([idx for idx in val_eps if idx not in evaled_list])
            global_idx = val_eps.index(ep_index)
            evaled_list.append(ep_index)
            start_idx = dataset_val.episode_data_index["from"][global_idx].item()
            end_idx = dataset_val.episode_data_index["to"][global_idx].item() - 1
            gt_ep_result = []
            pred_ep_result = []
            pred_ep_smoothed = []
            pred_ep_conf = []
            x_offset = 0
            # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
            eval_frame_gap = cfg.eval.eval_frame_gap
            smoother = RegressionConfidenceSmoother(value_range=(0.0, 1.0))
            print(f"[Eval Video] Evaluating episode_{ep_index}, progress: {i} / {cfg.eval.run_times}")

            # change to use tqdm
            for idx in tqdm(range(start_idx, end_idx, eval_frame_gap), desc=f"Processing episode {ep_index}"):
                data_point = dataset_val[idx]
                batch = adapt_lerobot_batch_sarm(data_point, camera_names=cfg.general.camera_names, eval_video=True)
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)

                # CLIP encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                # lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)
                if hasattr(img_emb, "pooler_output"):
                    img_emb = img_emb.pooler_output
                elif hasattr(img_emb, "last_hidden_state"):
                    img_emb = img_emb.last_hidden_state[:, 0]
                    
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)
                if hasattr(lang_emb, "pooler_output"):
                    lang_emb = lang_emb.pooler_output
                elif hasattr(lang_emb, "last_hidden_state"):
                    lang_emb = lang_emb.last_hidden_state[:, 0]

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)

                print(img_emb.shape, lang_emb.shape, state.shape)
                
                stage_prob = stage_model(img_emb, lang_emb, state, lens, scheme=model_type).softmax(dim=-1)  # (B, T, num_classes)
                stage_idx = stage_prob.argmax(dim=-1)  # (B, T)
                stage_conf = stage_prob.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)
                
                # Inject stage prior to subtask model
                stage_onehot = F.one_hot(stage_idx, num_classes=stage_prob.size(-1)).float()  # (B, T, C)
                stage_emb = stage_onehot.unsqueeze(1)                          # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
                
                pred = torch.clip(subtask_pred + stage_idx.float(), 0, num_classes-1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
        
                if model_type == "sparse":
                    raw_item_norm = normalize_sparse(raw_item)
                else:
                    raw_item_norm = normalize_dense(raw_item)
                
                conf_val = stage_conf[0, cfg.model.n_obs_steps].item()
                if idx >= (x_offset * eval_frame_gap):
                    smoothed_item = smoother.update(raw_item_norm, conf_val)
                else:
                    smoothed_item = raw_item_norm
                
                pred_ep_result.append(raw_item_norm)
                pred_ep_conf.append(conf_val)
                pred_ep_smoothed.append(smoothed_item)
                if dataset_type == "sparse":
                    gt_ep_result.append(normalize_sparse(trg[0, cfg.model.n_obs_steps].item()))
                else:
                    gt_ep_result.append(normalize_dense(trg[0, cfg.model.n_obs_steps].item()))

            # save results
            save_dir = plot_episode_result(ep_index, pred_ep_smoothed, gt_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_conf=pred_ep_conf, ep_smoothed=pred_ep_smoothed)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))
            np.save(Path(save_dir) / "gt.npy", np.array(gt_ep_result))
            np.save(Path(save_dir) / "smoothed.npy", np.array(pred_ep_smoothed))
            print(f"[Eval Video] episode_{ep_index} making video...")
            chunk_id = ep_index // 1000
            root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id # or change to your LEROBOT_LOCAL_DIR
            middle_video_dir = root / f"videos/chunk-{chunk_id:03d}/top_camera-images-rgb"
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}, progress: {i+1} / {cfg.eval.run_times}")

    
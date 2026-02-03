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
import wandb

from lerobot.common.datasets.rm_lerobot_dataset import FrameGapLeRobotDataset 
from utils.data_utils import get_valid_episodes, split_train_eval_episodes, adapt_lerobot_batch_rewind
from utils.train_utils import set_seed, save_ckpt, get_normalizer_from_calculated, plot_episode_result, plot_episode_result_raw_data
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data, normalize_dense
from models.rewind_reward_model import RewardTransformer
from models.clip_encoder import FrozenCLIPEncoder
from utils.make_demo_video import produce_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"


class ReWiNDWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def train(self):
        cfg = self.cfg
        OmegaConf.save(cfg, self.save_dir / "config.yaml")
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}-{cfg.general.task_name}',
            name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
            config=cfg,
        )

        # --- data ---
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        dataset_train = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=train_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               annotation_list=cfg.model.annotation_list,
                                               task_name=cfg.general.task_name)

        dataset_val = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               annotation_list=cfg.model.annotation_list,
                                               task_name=cfg.general.task_name)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512

        # --- reward_model ---
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        
        if cfg.model.resume_training:
            reward_model_path = Path(cfg.model.model_path)
            reward_ckpt = torch.load(reward_model_path, map_location=self.device)
            reward_model.load_state_dict(reward_ckpt["model"])
            reward_model.to(self.device)
            reward_model.train()

        # Optimizer
        reward_optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        
        # Schedulers
        reward_warmup_scheduler = LinearLR(
            reward_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        reward_cosine_scheduler = CosineAnnealingLR(
            reward_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,  
            eta_min=0.0
        )
        reward_scheduler = SequentialLR(
            reward_optimizer,
            schedulers=[reward_warmup_scheduler, reward_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        # ==================== training loop ==================================
        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            reward_model.train()
            with tqdm(dataloader_train, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    batch = adapt_lerobot_batch_rewind(batch, camera_names=cfg.general.camera_names)

                    B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                    img_list = []
                    for key in self.camera_names:
                        imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                        img_list.append(imgs)
                    
                    lang_strs = batch["tasks"]
                    trg = batch["targets"].to(self.device)
                    lens = batch["lengths"].to(self.device)
                    state = batch["state"].to(self.device)
                    
                    with torch.no_grad():
                        state = state_normalizer.normalize(state)
                        # CLIP encoding
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                    if cfg.model.no_state:
                        state = torch.zeros_like(state, device=self.device)
                    reward_pred = reward_model(img_emb, lang_emb, state, lens)
                    reward_loss = F.mse_loss(reward_pred, trg, reduction="mean")

                    reward_optimizer.zero_grad()
                    reward_loss.backward()
                    reward_unclipped = nn.utils.clip_grad_norm_(reward_model.parameters(), float("inf")).item()
                    _ = nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.train.grad_clip)
                    reward_optimizer.step()
                    reward_scheduler.step()
                    
                    if step % cfg.train.log_every == 0:
                        wandb.log({
                            "train/total_loss": reward_loss.item(),
                            "train/lr": reward_scheduler.get_last_lr()[0],
                            "train/reward_grad_norm": reward_unclipped,
                            "epoch": epoch,
                        }, step=step)
                    
                    pbar.set_postfix(loss=f"{(reward_loss.item()):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name=f"reward_step_{step:06d}_loss_{reward_loss.item():.3f}")
                    step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                reward_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = adapt_lerobot_batch_rewind(batch, camera_names=cfg.general.camera_names)
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
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                        if cfg.model.no_state:
                            state = torch.zeros_like(state, device=self.device)
                        reward_pred = reward_model(img_emb, lang_emb, state, lens)
                        reward_loss = F.mse_loss(reward_pred, trg, reduction="mean")
                        total_loss += reward_loss.item()
                        num += 1

                val_loss = total_loss / num 
                print(f"[Eval] Epoch {epoch} Val L1: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss}, step=step)

            torch.cuda.empty_cache()

            # --- save checkpoints ---
            save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_final")
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()

    # Evaluate whole trajectory from demo data, generating video
    def eval(self):
        import random
        cfg = self.cfg
        repo_id = cfg.general.repo_id
        valid_episodes = get_valid_episodes(repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)
        dataset_val = FrameGapLeRobotDataset(repo_id=repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               annotation_list=cfg.model.annotation_list,
                                               task_name=cfg.general.task_name,
                                               video_eval=True)
        
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names))
        

        # Load checkpoints
        reward_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.reward_model_name
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        reward_model.load_state_dict(reward_ckpt["model"])
        reward_model.to(self.device)
        reward_model.eval()

        # save path
        rollout_save_dir =  Path(self.save_dir) / "eval_video"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        evaled_list = []

        for i in range(cfg.eval.run_times):
            ep_index = random.choice([idx for idx in valid_episodes if idx not in evaled_list])
            global_idx = valid_episodes.index(ep_index)
            evaled_list.append(ep_index)
            start_idx = dataset_val.episode_data_index["from"][global_idx].item()
            end_idx = dataset_val.episode_data_index["to"][global_idx].item()
            gt_ep_result = []
            pred_ep_result = []
            pred_ep_smoothed = []
            x_offset = 0
            # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
            eval_frame_gap = cfg.eval.eval_frame_gap
            print(f"[Eval Video] Evaluating episode_{ep_index}, progress: {i} / {cfg.eval.run_times}")

            for idx in tqdm(range(start_idx, end_idx, eval_frame_gap), desc=f"Processing episode {ep_index}"):
                data_point = dataset_val[idx]
                batch = adapt_lerobot_batch_rewind(data_point, camera_names=cfg.general.camera_names, eval_video=True)
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
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                
                reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                smoothed_item = raw_item
                
                pred_ep_result.append(raw_item)
                gt_ep_result.append(normalize_dense(trg[0, cfg.model.n_obs_steps].item()))
                pred_ep_smoothed.append(smoothed_item)
                
            # save results
            save_dir = plot_episode_result(ep_index, pred_ep_smoothed, gt_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap)
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


    def eval_raw_data(self):
        import random
        cfg = self.cfg
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoding
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  )

        # Load checkpoints
        reward_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.reward_model_name
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        reward_model.load_state_dict(reward_ckpt["model"])
        reward_model.to(self.device)
        reward_model.eval()

        # Eval save path
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        rollout_save_dir =  Path(self.save_dir) / "eval_video" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")

        
        x_offset = 0
        # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
        data_dir = cfg.eval.raw_data_dir
        run_times = cfg.eval.raw_data_run_times
        # Get all valid episode paths
        all_episodes = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("episode_")
        ]
        eval_list = all_episodes
        
        random.seed(cfg.general.seed)
        # randomly select eval_list
        if len(all_episodes) >= run_times:
            eval_list = random.sample(all_episodes, run_times)
        else:
            raise ValueError(f"Not enough episodes in {data_dir} to sample {run_times} items.")


        for i in range(run_times):
            data_path = eval_list[i]
            pred_ep_result = []
            pred_ep_smoothed = []
            # randomly select 
            ep_index = os.path.basename(data_path)
            frame_num = get_frame_num(data_path)
            traj_joint_data = get_traj_data(data_path)
            eval_frame_gap = cfg.eval.eval_frame_gap
            print(f"[EVAL_RAW]: process {i+1}/{run_times} episode: {ep_index}")
            for idx in tqdm(range(0, frame_num, eval_frame_gap), desc=f"Processing data"):
                batch = get_frame_data_fast(path=data_path, 
                                    traj_joint_data=traj_joint_data, 
                                    idx=idx,
                                    n_obs_steps=cfg.model.n_obs_steps,
                                    frame_gap=cfg.model.frame_gap,
                                    max_rewind_steps=cfg.model.max_rewind_steps,
                                    camera_names=cfg.general.camera_names,
                                    device=self.device)
                
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = ["fold the tshirt"]
                lens = torch.tensor([1+cfg.model.n_obs_steps], dtype=torch.int32, device=self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)
                
                # CLIP encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                smoothed_item = raw_item
                
                pred_ep_result.append(raw_item)
                pred_ep_smoothed.append(smoothed_item)
                

            # save results
            save_dir = plot_episode_result_raw_data(ep_index, pred_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_smoothed=None)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))

            print(f"[Eval Video] episode_{ep_index} making video...")
            middle_video_dir = Path(f"{data_path}/top_camera-images-rgb.mp4")
            
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              raw_data=True,
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}")


    
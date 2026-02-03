import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
from typing import List, Optional, Union

def produce_video(
    save_dir: Union[str, Path],
    middle_video: Union[str, Path],
    episode_num: Optional[int] = None,
    *,
    raw_data: bool = False,
    x_offset: int = 30,
    frame_gap: Optional[int] = None,
    frame_rate: int = 30,
    target_h: int = 448,
    target_w: int = 448,
):
    """
    Create a side-by-side video panel with the middle camera view and a progress plot.

    Two modes (controlled by `raw_data`):
      - raw_data=False (episode mode): expects per-episode folder structure:
            save_dir/episode_{episode_num}/pred.npy
            save_dir/episode_{episode_num}/conf.npy (optional)
            save_dir/episode_{episode_num}/smoothed.npy (optional)
            save_dir/episode_{episode_num}/gt.npy (optional)
        and a middle video at:
            middle_video_dir/episode_{episode_num:06d}.mp4
        where `middle_video` is the directory containing the per-episode .mp4 files.

      - raw_data=True (raw-data mode): expects files directly in `save_dir`:
            save_dir/pred.npy
            save_dir/conf.npy (optional)
            save_dir/smoothed.npy (optional)
        and `middle_video` is the full path to the video file.

    Args:
        save_dir: Root directory for saving / reading arrays.
        middle_video: Either a directory (episode mode) or a full video path (raw-data mode).
        episode_num: Episode index (required if raw_data=False).
        raw_data: Switch between episode mode (False) and raw-data mode (True).
        x_offset: Number of initial prediction steps to skip.
        frame_gap: If provided, skip frames from the video by this factor when applying x_offset.
        frame_rate: FPS used when reading frames and writing output.
        target_h, target_w: Per-panel resolution.

    Output:
        Writes a combined video to:
          - episode mode: save_dir/episode_{episode_num}/combined_video.mp4
          - raw-data mode: save_dir/combined_video.mp4
    """
    save_dir = Path(save_dir)

    # -------------------------
    # Resolve paths per mode
    # -------------------------
    if raw_data:
        episode_dir = save_dir / episode_num
        middle_video_path = Path(middle_video)
        output_path = episode_dir / "combined_video.mp4"
        
        pred_path   = episode_dir / "pred.npy"
        conf_path   = episode_dir / "conf.npy"
        smooth_path = episode_dir / "smoothed.npy"
        gt_path     = episode_dir / "gt.npy"  # optional; not required in raw mode
    else:
        if episode_num is None:
            raise ValueError("`episode_num` must be provided when raw_data=False.")
        episode_dir = save_dir / f"episode_{episode_num}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        middle_video_dir = Path(middle_video)
        middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
        output_path = episode_dir / "combined_video.mp4"

        pred_path   = episode_dir / "pred.npy"
        conf_path   = episode_dir / "conf.npy"
        smooth_path = episode_dir / "smoothed.npy"
        gt_path     = episode_dir / "gt.npy"  # optional

    # -------------------------
    # Load arrays
    # -------------------------
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    pred_full = np.load(pred_path)
    pred = pred_full[x_offset:]

    # confidence (optional)
    conf = None
    if conf_path.exists():
        conf = np.load(conf_path)[x_offset:]

    # smoothed (optional, fallback to pred)
    if smooth_path.exists():
        smoothed = np.load(smooth_path)[x_offset:]
    else:
        smoothed = pred

    # ground truth (optional; used only in episode mode originally, but we just load if present)
    gt = None
    if gt_path.exists():
        gt_full = np.load(gt_path)
        gt = gt_full[x_offset:]

    T = len(pred)

    # -------------------------
    # Load and align video frames
    # -------------------------
    if not middle_video_path.exists():
        raise FileNotFoundError(f"Missing video file: {middle_video_path}")

    clip_middle = VideoFileClip(str(middle_video_path))
    raw_frames = list(clip_middle.iter_frames(fps=frame_rate))

    # Apply x_offset at the frame level (optionally scaled by frame_gap)
    start_idx = x_offset * frame_gap if frame_gap is not None else x_offset
    if start_idx >= len(raw_frames):
        raise ValueError(
            f"x_offset ({x_offset}) with frame_gap ({frame_gap}) exceeds video length ({len(raw_frames)})."
        )
    frames_middle = raw_frames[start_idx:]

    # Align prediction length with available frames
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(
            f"WARNING: Not enough frames in video. Expected {T}, found {min_frames_num}. "
            f"Truncating predictions by {gap} to match video."
        )
        T = min_frames_num
        pred = pred[gap:]
        if conf is not None:
            conf = conf[gap:]
        if smoothed is not None:
            smoothed = smoothed[gap:]
        if gt is not None:
            gt = gt[gap:]

    # Uniformly sample/align frames to match T
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]

    # -------------------------
    # Compose panels
    # -------------------------
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame(
            t,
            pred,
            x_offset,
            height=target_h,
            width=target_w,
            frame_gap=frame_gap,
            smoothed=smoothed,
        )
        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # -------------------------
    # Write video
    # -------------------------
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec="libx264")


def draw_plot_frame(step: int, pred, x_offset, width=448, height=448, frame_gap=None, smoothed=None):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    if frame_gap is None:
        timesteps = np.arange(len(pred)) + x_offset
    else:
        timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap
        
    # === Plot raw prediction ===
    pred = smoothed if smoothed is not None else pred
    line_pred, = ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    handles, labels = [line_pred], ["Predicted"]

    # === Vertical line at current step ===
    if frame_gap is None:
        ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=step * frame_gap + x_offset * frame_gap, color='r', linestyle='--', linewidth=2)

    # === Labels and style ===
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # === Legend ===
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img



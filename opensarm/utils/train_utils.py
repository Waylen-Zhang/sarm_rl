import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.normalizer import SingleFieldLinearNormalizer
import json
import numpy as np

def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_ckpt(model, opt, ep, save_dir, input_name=None):
    save_dir = Path(save_dir) / "checkpoints"  # convert to Path first
    name = f"{input_name}.pt" if input_name else f"epoch{ep:04d}.pt"
    p = save_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(model=model.state_dict(),
             optimizer=opt.state_dict(),
             epoch=ep),
        p
    )

@torch.no_grad()
def get_normalizer_from_calculated(path, device) -> "SingleFieldLinearNormalizer":
    """
    Load norm stats from a JSON file. Accepts absolute or relative paths.
    Relative paths are resolved robustly even when Hydra changes the CWD.
    """
    # --- Resolve path robustly ---
    original = str(path)
    abs_path = None

    # 1) Try Hydra's original working dir resolution (best for Hydra runs)
    try:
        from hydra.utils import to_absolute_path
        candidate = Path(to_absolute_path(original))
        if candidate.exists():
            abs_path = candidate
    except Exception:
        pass

    # 2) If already absolute and exists, use it
    if abs_path is None:
        candidate = Path(original)
        if candidate.is_absolute() and candidate.exists():
            abs_path = candidate

    # 3) Try common relative bases: current CWD, this file's dir, and a couple of parents
    if abs_path is None and not Path(original).is_absolute():
        bases = []
        try:
            here = Path(__file__).resolve().parent
            bases.extend([Path.cwd(), here, here.parent, here.parent.parent])
        except Exception:
            bases.append(Path.cwd())

        for base in bases:
            candidate = base / original
            if candidate.exists():
                abs_path = candidate
                break

    if abs_path is None:
        tried = [f"- Hydra to_absolute_path('{original}')",
                 f"- Absolute '{original}'" if Path(original).is_absolute() else f"- '{Path.cwd() / original}' (CWD)",
                 "- __file__/.. variations"]
        raise FileNotFoundError(
            f"Could not locate normalizer JSON '{original}'. Tried:\n" + "\n".join(tried)
        )

    # --- Load and build normalizer ---
    with open(abs_path, "r") as f:
        norm_data = json.load(f)["norm_stats"]

    def to_tensor_slice(data, k: int = 14):  # both arms
        return torch.tensor(data[:k], dtype=torch.float32, device=device)

    state_stats = norm_data["state"]
    state_std = to_tensor_slice(state_stats["std"])
    state_mean = to_tensor_slice(state_stats["mean"])

    state_normalizer = SingleFieldLinearNormalizer.create_manual(
        scale=1.0 / state_std,
        offset=-(state_mean / state_std),
        input_stats_dict={
            "min": to_tensor_slice(state_stats["q01"]),
            "max": to_tensor_slice(state_stats["q99"]),
            "mean": state_mean,
            "std": state_std,
        },
    )
    return state_normalizer


def plot_episode_result(ep_index, ep_result, gt_ep_result, x_offset, rollout_save_dir, frame_gap=None, ep_conf=None, ep_smoothed=None):
    save_dir = rollout_save_dir / f"episode_{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Trim initial frames
    ep_result = ep_result[x_offset:]
    gt_ep_result = gt_ep_result[x_offset:]

    # Convert to numpy arrays
    ep_result_np = np.array(ep_result)
    gt_ep_result_np = np.array(gt_ep_result)
    ep_conf_np = np.asarray(ep_conf)[x_offset:] if ep_conf is not None else None
    ep_smoothed_np = np.asarray(ep_smoothed)[x_offset:] if ep_smoothed is not None else None
    if ep_smoothed_np is not None:
        ep_result_np = ep_smoothed_np 
    
    # === Timesteps ===
    if frame_gap is None:
        timesteps = np.arange(len(ep_result_np)) + x_offset
    else:
        timesteps = np.arange(0, len(ep_result_np) * frame_gap, frame_gap) + x_offset * frame_gap

    # Compute MSE and MAE
    mse = np.mean((ep_result_np - gt_ep_result_np) ** 2)
    mae = np.mean(np.abs(ep_result_np - gt_ep_result_np))

    # Plot
    plt.figure()
    plt.plot(timesteps, ep_result_np, label="Predicted")
    plt.plot(timesteps, gt_ep_result_np, label="GT")
    if ep_conf_np is not None:
        plt.plot(timesteps, ep_conf_np, label="Conf", linestyle=":", color="green")
    # Add dummy lines for metrics in the legend
    plt.plot([], [], ' ', label=f"MSE: {mse:.4f}")
    plt.plot([], [], ' ', label=f"MAE: {mae:.4f}")
    plt.title("Episode Result")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "plot.png")
    plt.close()

    return str(save_dir)


def plot_episode_result_raw_data(ep_index, ep_result, x_offset, rollout_save_dir, frame_gap=None, ep_conf=None, ep_smoothed=None):
    save_dir = rollout_save_dir / f"{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Trim & to numpy ===
    ep_result_np = np.asarray(ep_result)[x_offset:]  # raw predictions
    ep_conf_np = np.asarray(ep_conf)[x_offset:] if ep_conf is not None else None
    ep_smoothed_np = np.asarray(ep_smoothed)[x_offset:] if ep_smoothed is not None else None

    # === Handle empty results ===
    if len(ep_result_np) == 0:
        print(f"Warning: Episode {ep_index} has no results after trimming. Skipping plot.")
        return None

    # === Timesteps ===
    if frame_gap is None:
        timesteps = np.arange(len(ep_result_np)) + x_offset
    else:
        timesteps = np.arange(0, len(ep_result_np) * frame_gap, frame_gap) + x_offset * frame_gap

    # === Plot ===
    fig, ax = plt.subplots(figsize=(4.48, 4.48), dpi=100)  # 448x448 px
    line_pred, = ax.plot(timesteps, ep_result_np, label="Raw Predicted", linewidth=2)
    handles, labels = [line_pred], ["Raw Predicted"]

    if ep_smoothed_np is not None:
        line_smooth, = ax.plot(timesteps, ep_smoothed_np, label="Smoothed", linewidth=2, color="orange")
        handles.append(line_smooth)
        labels.append("Smoothed")

    ax.set_title("Episode Result")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Prediction")
    ax.grid(True)

    # Confidence on twin y-axis
    if ep_conf_np is not None:
        ax2 = ax.twinx()
        line_conf, = ax2.plot(timesteps, ep_conf_np, linestyle=":", label="Confidence", color="green")
        ax2.set_ylabel("Confidence")
        # Merge legends
        handles.append(line_conf)
        labels.append("Confidence")

    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    out_path = save_dir / "plot.png"
    fig.savefig(out_path)
    plt.close(fig)

    return str(save_dir)
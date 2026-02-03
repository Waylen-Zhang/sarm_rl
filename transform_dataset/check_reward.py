import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/dx/waylen/SARM/transformed_datasets/insert_wire")

DATA_PARQUET = DATASET_PATH / "data" / "chunk-000" / "file-000.parquet"
EPISODE_PARQUET = DATASET_PATH / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

def main():
    print("Loading frame data...")
    df = pd.read_parquet(DATA_PARQUET)

    print("Loading episode metadata...")
    ep = pd.read_parquet(EPISODE_PARQUET)

    print("\n========== Basic Info ==========")
    print("Total frames:", len(df))
    print("Episodes:", len(ep))

    rewards = df["next.reward"].apply(lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x))

    print("\n========== Reward Stats ==========")
    print("Min:", rewards.min())
    print("Max:", rewards.max())
    print("Mean:", rewards.mean())
    print("Std:", rewards.std())

    print("\nNaN count:", rewards.isna().sum())
    print("Zero count:", (rewards == 0).sum())

    print("\n========== Per-Episode Check ==========")

    for i, row in ep.iterrows():
        start = row["start_frame"]
        end = row["end_frame"]

        ep_rewards = rewards.iloc[start:end+1].values

        print(f"\nEpisode {i}: frames={len(ep_rewards)}")
        print("  first reward:", ep_rewards[0])
        print("  last reward :", ep_rewards[-1])
        print("  monotonic increasing:", np.all(np.diff(ep_rewards) >= -1e-6))

        if np.isnan(ep_rewards).any():
            print("  ❌ Contains NaN")

        if np.all(ep_rewards == 0):
            print("  ⚠️ All zero rewards")

        if ep_rewards[-1] < 0.9:
            print("  ⚠️ Final reward too small")

if __name__ == "__main__":
    main()

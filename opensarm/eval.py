import sys
import argparse
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

# Parse --mode before Hydra sees sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="eval", choices=["eval", "raw_data"])
args, remaining_argv = parser.parse_known_args()

# Replace sys.argv with only the remaining args for Hydra
sys.argv = [sys.argv[0]] + remaining_argv

@hydra.main(config_path="config", config_name=None, version_base="1.1")
def main(cfg: DictConfig):
    workspace = instantiate(cfg)

    if args.mode == "eval":
        workspace.eval()
    elif args.mode == "raw_data":
        workspace.eval_raw_data()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()

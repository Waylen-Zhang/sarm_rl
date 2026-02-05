import sys
import argparse
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import rclpy


# ---------- parse --mode before hydra ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="ros",
    choices=["ros"],
    help="Deployment mode"
)
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv


@hydra.main(config_path="config", config_name=None, version_base="1.1")
def main(cfg: DictConfig):
    rclpy.init()
    workspace = instantiate(cfg)

    if args.mode == "ros":
        workspace.deploy()
    else:
        raise ValueError(f"Unsupported deploy mode: {args.mode}")
    try:
        rclpy.spin(workspace.node)
    except KeyboardInterrupt:
        pass
    finally:
        workspace.node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

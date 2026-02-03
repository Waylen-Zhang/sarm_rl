import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

@hydra.main(config_path="config", config_name=None)
def main(cfg: DictConfig):
    workspace = instantiate(cfg)
    workspace.train()

if __name__ == "__main__":
    main()

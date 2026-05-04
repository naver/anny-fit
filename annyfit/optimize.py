import os
import argparse
import torch
from omegaconf import DictConfig, OmegaConf

from annyfit import Annyfit


def main(cfg: DictConfig, args):
    os.makedirs(cfg.logger.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annyfit = Annyfit(cfg, device=device)
    annyfit.run(valid_images=args.valid_images)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description="Run Anyfits on a dataset")
    argparse.add_argument("--config", type=str, default="configs/demo/multihmr.yaml", help="Path to the configuration file")
    argparse.add_argument("--valid_images", nargs='+', required=False, default=None, help="Optional: A list of specific image files to process.")

    args = argparse.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args)

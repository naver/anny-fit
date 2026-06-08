import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from annyfit_stage import AnnyfitStage

class AnnyfitMultiPerson(torch.nn.Module):
    def __init__(self, cfg: DictConfig, logger, initial_params, target_data, img_path, camera_intrinsics, device):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.initial_params = initial_params
        self.target_data = target_data
        self.img_path = img_path
        self.camera_intrinsics = camera_intrinsics
        self.device = device

    def get_final_vertices(self):
        return self.annyfit_stage.get_final_vertices()

    def get_initial_vertices(self):
        return self.annyfit_stage.get_initial_vertices()

    def optimize(self):
        """
        Multi stage optimization for all people in an image.
        """
        print(f"Running multi-stage optimization for {self.img_path}")
        self.annyfit_stage = AnnyfitStage(
            target=self.target_data,
            initial_params=self.initial_params,
            cfg=self.cfg,
            img_path=self.img_path,
            logger=self.logger,
            K=self.camera_intrinsics,
        )

        self.annyfit_stage.to(self.device)

        # per stage optimization loop
        global_step = 0
        for stage_idx, stage in enumerate(self.cfg.stages):
            global_step = self.annyfit_stage.optimize_stage(stage, stage_idx, global_step)

        # collect the final optimized parameters
        final_params = self.annyfit_stage.get_final_params()

        return final_params

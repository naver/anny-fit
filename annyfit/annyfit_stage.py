
import os
import torch
import cv2
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from anny_wrapper import Anny, MyParameterDict
from utils import perspective_projection, get_camera_parameters, solvePnP
from render import visualize_and_save, visualize_points
from losses import BodyFittingLoss


from PIL import Image, ImageOps
import numpy as np

class AnnyfitStage(pl.LightningModule):
    """
    AnnyfitStage: Optimizes an Anny model instance by fitting it to 2D keypoints.
    """
    def __init__(self, target, cfg: DictConfig, initial_params: dict,  img_path: str, logger, K: torch.Tensor=None):
        super().__init__()
        # saves the config and creates self.hparams
        self.cfg = cfg
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        self.save_hyperparameters(cfg_dict)
        # init the model
        self.num_people = initial_params['root_translation_params'].shape[0]
        self.anny_model = Anny(batch_size=self.num_people)
        self.init_target(target)
        self.anny_model.init_parameters(initial_params)

        self.img_path = img_path
        self.img_name = os.path.basename(img_path)
        self.exp_logger = logger
        self.vis_dir = os.path.join(self.exp_logger.log_dir, 'vis')
        os.makedirs(self.vis_dir, exist_ok=True)

        self.img = cv2.imread(self.img_path)
        img_size = self.img.shape[:2] # (height, width)

        if K is None:
            print("Camera matrix K not provided. Initializing with default values.")
            K = get_camera_parameters(
                img_size=img_size,
                fov=self.cfg.camera.fov,
                p_x=0.5,
                p_y=0.5
            )
            K = K.unsqueeze(0)  # add batch dimension

        self.faces = self.anny_model.faces

        self.register_buffer('K', K)

        self.fitting_loss = BodyFittingLoss(loss_cfg=self.cfg.loss)

    def init_target(self, target):
        self.target = TargetData()
        self.target.set_data('keypoints_2d', target['keypoints_2d'])
        if 'dense_kp' in target:
            self.target.set_data('dense_kp', target['dense_kp'])

        if 'depth' in target:
            self.target.set_data('depth', target['depth'])

        if 'keypoints_2d_depth' in target:
            self.target.set_data('keypoints_2d_depth', target['keypoints_2d_depth'])

        if 'shape_attributes' in target:
            shape_attr = target['shape_attributes']
            indices = shape_attr.get('attr_indices', [])
            if len(indices) > 0:
                self.target.set_data('shape_attr_batch_indices', torch.tensor(shape_attr['batch_indices'], dtype=torch.long))
                self.target.set_data('shape_attr_indices', torch.tensor(indices, dtype=torch.long))
                self.target.set_data('shape_attr_values', torch.tensor(shape_attr['attr_values'], dtype=torch.float32))

        if 'masks' in target:
            self.target.set_data('masks', target['masks'])

        if 'depth_map' in target:
            self.target.set_data('depth_map', target['depth_map'])

    def visualize_mesh(self, vertices, save_path, keypoints=None):
        vis_img = self.img.copy()
        try:
            if keypoints is not None:
                vis_img = visualize_points(vis_img, keypoints, save_path, color=(0, 0, 255))
                # add target points
                vis_img = visualize_points(vis_img, self.target.keypoints_2d[:, :, :2].clone().detach().cpu(), save_path, color=(0, 255, 0))
        except:
            print("Error visualizing keypoints.")
        visualize_and_save(vis_img, vertices, self.faces, self.K, save_path)

    def get_ignore_params(self, stage):
        ignore_params = []
        if "fingertoe" in stage.ignore_params:
            ignore_params += list(self.anny_model.fingertoe_params_names)
        if "face" in stage.ignore_params:
            ignore_params += list(self.anny_model.face_params_names)
        return ignore_params

    def _unpack_parameters(self, param_container, ignore_params=None):
        if ignore_params is None:
            ignore_params = set()

        # case 1: custom container with nested parameters
        if isinstance(param_container, MyParameterDict):
            return [p for name, p in param_container.items() if name not in ignore_params]

        # case 2: a PyTorch ParameterDict
        if isinstance(param_container, torch.nn.ParameterDict):
            return list(param_container.values())

        # case 3: a single parameter tensor
        if isinstance(param_container, torch.nn.Parameter):
            return [param_container]

        return []

    def init_optimizer(self, stage):
        """Creates a new optimizer for a given stage."""
        # freeze all parameters to reset state
        for param in self.parameters():
            param.requires_grad = False

        params_for_stage = []
        lr_config = self.cfg.optimizer.param_groups

        ignore_params = self.get_ignore_params(stage)

        for param_name in stage.params:
            param_container = getattr(self.anny_model, param_name)

            # get individual tensors from nested parameter containers
            unpacked_params = self._unpack_parameters(param_container, ignore_params)

            # unfreeze the unpacked parameters
            for p in unpacked_params:
                p.requires_grad = True

            # assign learning rate for the group
            lr_key = f"{param_name}_lr"
            lr = lr_config.get(lr_key, self.cfg.optimizer.lr)
            params_for_stage.append({'params': unpacked_params, 'lr': lr})

        if not params_for_stage:
            raise ValueError(f"Optimizer has no parameters for stage. Check stage config: {stage.params}")

        return torch.optim.Adam(params_for_stage)

    def log_step(self, losses, global_step, stage_idx, stage):
        if not hasattr(self.exp_logger, 'experiment'):
            return
        metrics_to_log = losses.copy()
        metrics_to_log['stage'] = float(stage_idx)
        for key, value in metrics_to_log.items():
            if isinstance(value, torch.Tensor):
                continue
            self.exp_logger.experiment.add_scalar(key, value, global_step=global_step)

    def get_final_params(self):
        pose_keys, pose_values = self.anny_model.get_bodypose_parameters()
        shape_keys, shape_values = self.anny_model.get_shape_parameters()
        local_keys, local_values = self.anny_model.get_local_changes_parameters()
        final_params = {
            'root_rotation_params': self.anny_model.root_rotation_params.clone().detach().unsqueeze(1), # (bs, 1, 3)
            'root_translation_params': self.anny_model.root_translation_params.clone().detach(), # (bs, 3)
            'body_pose_params': pose_values.clone().detach(), # (bs, 162, 3)
            'shape_params': shape_values.clone().detach(), # (bs, 11)
            'local_changes_params': local_values.clone().detach(), # (bs, 256)
            'body_pose_keys': [pose_keys] * self.num_people,  # (bs, 162)
            'shape_keys': [shape_keys] * self.num_people,  # (bs, 11)
            'local_changes_keys': [local_keys] * self.num_people,  # (bs, 256)
            'camera_intrinsics': self.K.clone().detach().repeat(self.num_people, 1, 1),  # (bs, 3, 3)
        }

        return final_params

    def get_final_vertices(self):
        return self.anny_model.verts_init.clone().detach()  # (bs, V, 3)

    def get_initial_vertices(self):
        return self.initial_vertices  # (bs, V, 3)

    def _perform_one_optimization_step(self, optimizer, step_idx):
        optimizer.zero_grad()
        # forward through the body model
        anny_output = self.anny_model.forward()

        # project 3D points to 2D
        coco_joints = anny_output['coco_joints']  # (bs, 17, 3)
        est_kpts_2d = perspective_projection(coco_joints, self.K)
        est_dense_kpts_2d = perspective_projection(anny_output['dense_joints'], self.K)

        # estimate the shape attributes
        est_shape_attr = anny_output['shape'][self.target.shape_attr_batch_indices, self.target.shape_attr_indices]

        est_depth_scale = anny_output['depth_scale']
        est_depth_shift = anny_output['depth_shift']

        # Calculate losses
        total_loss, loss_dict = self.fitting_loss(
            model_verts=anny_output['vertices'],
            body_pose=anny_output['body_pose'],
            shape=anny_output['shape'],
            est_kpts_2d=est_kpts_2d,
            est_dense_kp=est_dense_kpts_2d,
            verts_init=self.anny_model.verts_init,
            init_pose=self.anny_model.init_pose,
            init_shape=self.anny_model.init_shape,
            target_kpts_2d=self.target.keypoints_2d,
            target_dense_kp=self.target.dense_kp,
            est_shape_attr=est_shape_attr,
            est_depth=anny_output['scaled_depth'],
            est_kp_depth=anny_output['scaled_kp_depth'],
            est_depth_scale=est_depth_scale,
            est_depth_shift=est_depth_shift,
            target_shape_attr=self.target.shape_attr_values,
            target_depth=self.target.depth,
            target_kp_depth=self.target.keypoints_2d_depth,
        )
        total_loss.backward()
        optimizer.step()

        return loss_dict

    def optimize_stage(self, stage, stage_idx, global_step):
        """Optimizes the model for a single stage for the required number of epochs."""
        self.fitting_loss.update_weights(stage.loss_weights)
        optimizer = self.init_optimizer(stage)
        self.img_prefix = f"{self.img_name[:-4]}_{stage_idx}"

        if stage_idx == 0:
            self.initial_vertices = self.anny_model.verts_init.clone().detach()  # (bs, V, 3)

        for step_idx in tqdm(range(stage.epochs), desc=f"Stage {stage.name}"):
            losses = self._perform_one_optimization_step(optimizer, step_idx)
            self.log_step(losses, global_step, stage_idx, stage)
            global_step += 1

        # update the initial parameters for the next stage
        self.anny_model.update_init_parameters()

        # save final mesh
        save_path = os.path.join(self.vis_dir, f"{self.img_prefix}_final.jpg")
        self.visualize_mesh(vertices=self.anny_model.verts_init, save_path=save_path)

        return global_step


class TargetData(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('keypoints_2d', torch.tensor([]))
        self.register_buffer('dense_kp', torch.tensor([]))
        self.register_buffer('shape_attr_batch_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('shape_attr_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('shape_attr_values', torch.tensor([]))
        self.register_buffer('depth', torch.tensor([]))
        self.register_buffer('keypoints_2d_depth', torch.tensor([]))
        self.register_buffer('masks', torch.tensor([]))
        self.register_buffer('depth_map', torch.tensor([]))

    def set_data(self, key: str, value: torch.Tensor):
        setattr(self, key, value)

import numpy as np
import torch
import pickle
import roma
from dataclasses import dataclass
from collections import OrderedDict

import anny
from utils import clamp_but_preserve_gradients

from constants import SMPL_MODEL_DIR, SMPL2COCO_REGRESSOR, SMPLX2SMPL_REGRESSOR, SMPL2DENSE_REGRESSOR
from smplx.lbs import vertices2joints

class MyParameterDict(torch.nn.Module):
    """
    Dictionary of parameters that maps '.' into '_' to overcome PyTorch limitations.
    Useful for storing parameters where original keys include dots, such as joint names.
    """
    def __init__(self, params_tuple):
        super().__init__()
        self.key_mapping = OrderedDict()
        params = []
        self.param_names = []
        for (key, value) in params_tuple:
            valid_key = key.replace(".", "_")
            self.key_mapping[key] = valid_key
            params.append((valid_key, value))
            self.param_names.append(valid_key)
        assert len(set(self.key_mapping.values())) == len(self.key_mapping), "Collision during key mapping!"
        self.params = torch.nn.ParameterDict(params)

    def __getitem__(self, key):
        return self.params[self.key_mapping[key]]
    
    def keys(self):
        return self.key_mapping.keys()
    
    def param_keys(self):
        return self.param_names

    def values(self):
        return self.params.values()

    def items(self):
        for key in self.keys():
            yield key, self.__getitem__(key)

    def index(self, key):
        return self.param_names.index(self.key_mapping[key])

class Anny(torch.nn.Module):
    """ Extension of the Anny body model to support more joints and handle parameterization """

    def __init__(self, dtype=torch.float32, skinning_method='lbs', batch_size=1):
        super().__init__()
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = anny.create_fullbody_model(remove_unattached_vertices=False,
                                                local_changes=True,
                                                default_pose_parameterization='root_relative_world',
                                                topology='smplx',
                                                ).to(dtype=self.dtype)
        self.model.set_skinning_method(skinning_method)
        self.faces = self.model.get_triangular_faces()
        
        # joint limit ranges
        joint_limits = [
            (label, torch.nn.Parameter(
                torch.deg2rad(torch.stack([torch.as_tensor([-180., 180.]) for axis in 'xyz'], dim=-1)),
                requires_grad=False  # Tell PyTorch not to track gradients for these
            ))
            for label in self.model.bone_labels if label != "root"
        ]
        
        self.joint_limit_ranges = MyParameterDict(joint_limits)

        self.register_buffer('identity_rotation', torch.zeros(1, 3, dtype=dtype))
        self.register_buffer('null_translation', torch.zeros(1, 3, dtype=dtype))

        self.root_rotation_params = torch.nn.Parameter(torch.zeros(self.batch_size, 3, dtype=dtype).contiguous().requires_grad_(True))
        self.root_translation_params = torch.nn.Parameter(torch.zeros(self.batch_size, 1, 3, dtype=dtype, requires_grad=True))
        self.joints_rotation_params = MyParameterDict([(label, torch.zeros(self.batch_size, 3, dtype=dtype, requires_grad=True)) for label in self.joint_limit_ranges.keys()])
        # to maintain order of the parameters use list of tuples
        self.shape_params = torch.nn.ParameterDict([(key, torch.nn.Parameter(torch.full((self.batch_size,), fill_value=0.5, dtype=dtype, requires_grad=True))) for key in self.model.phenotype_labels])
        # not optimizing the local changes
        self.local_changes_kwargs = torch.nn.ParameterDict([(key, torch.nn.Parameter(torch.zeros(self.batch_size, dtype=dtype, requires_grad=False))) for key in self.model.local_change_labels]) 

        # SMPLX → SMPL vertex conversion
        with open(SMPLX2SMPL_REGRESSOR, 'rb') as f:
            smplx2smpl_data = pickle.load(f)
        self.register_buffer('smplx2smpl', torch.tensor(smplx2smpl_data['matrix'], dtype=self.dtype))  # (6890, 10475)

        # SMPL → COCO joints
        self.register_buffer('smpl2coco', torch.tensor(np.load(SMPL2COCO_REGRESSOR), dtype=self.dtype))

        # SMPL → 138 dense keypoints
        with open(SMPL2DENSE_REGRESSOR, 'rb') as f:
            dense_mat = pickle.load(f)
        if hasattr(dense_mat, 'to_dense'):
            dense_mat = dense_mat.to_dense()
        self.register_buffer('smpl2dense', dense_mat.to(dtype=self.dtype))  # (138, 6890)

        self.face_params_names = self.get_face_parameter_names()
        self.fingertoe_params_names = self.get_fingertoe_parameter_names()

        # parameters for depth scaling
        self.depth_params = torch.nn.ParameterDict({
            'depth_scale': torch.nn.Parameter(torch.tensor(1.0, dtype=self.dtype), requires_grad=True),
            'depth_shift': torch.nn.Parameter(torch.tensor(0.0, dtype=self.dtype), requires_grad=True)
        })

        
    def update_parameters(self, root_rotation_params=None, root_translation_params=None, body_pose_params=None, shape_params=None):
        """
        Updates the parameters of the model.
        Args:
            root_rotation_params (torch.Tensor): Rotation parameters for the root joint.
            root_translation_params (torch.Tensor): Translation parameters for the root joint.
            body_pose_params (dict): Body pose parameters for each joint.
            shape_params (dict): Shape parameters for macrodetails.
        """
        if root_rotation_params is not None:
            self.root_rotation_params.data = root_rotation_params.to(dtype=self.dtype)
        if root_translation_params is not None:
            self.root_translation_params.data = root_translation_params.to(dtype=self.dtype)
        if body_pose_params is not None:
            for idx, label in enumerate(self.joints_rotation_params.keys()):
                self.joints_rotation_params[label].data = body_pose_params[:, idx].to(dtype=self.dtype)
        if shape_params is not None:
            for idx, key in enumerate(self.shape_params.keys()):
                self.shape_params[key].data = shape_params[:, idx].to(dtype=self.dtype)

    def init_parameters(self, init_parameters):
        """
        Initializes the parameters of the model.
        Args:
            init_parameters (dict): Dictionary containing initial parameters for the model.
        """

        self.update_parameters(
            root_rotation_params=init_parameters.get('root_rotation_params', None),
            root_translation_params=init_parameters.get('root_translation_params', None),
            body_pose_params=init_parameters.get('body_pose_params', None),
            shape_params=init_parameters.get('shape_params', None)
        )
        # save the initial parameters for the loss
        self.init_pose = torch.nn.Parameter(torch.stack(list(self.joints_rotation_params.values()), dim=1))  # (bs, 162, 3)
        self.init_shape = torch.nn.Parameter(torch.stack(list(self.shape_params.values()), dim=1))  # (bs, 8)
        with torch.no_grad():
            v3d = self.forward()['vertices']

        self.verts_init = torch.nn.Parameter(v3d)

    def update_init_parameters(self):
        """
        Use a clone of the current parameters to update the initial parameters.
        """
        self.init_pose.data = torch.stack(list(self.joints_rotation_params.values()), dim=1).clone().detach()
        self.init_shape.data = torch.stack(list(self.shape_params.values()), dim=1).clone().detach()
        with torch.no_grad():
            v3d = self.forward()['vertices']
        self.verts_init.data = v3d.clone().detach()

    def get_face_parameter_names(self):
        """
        Returns the names of the parameters used for the face.
        """
        face_prefixes = ["oris", "levator", "eye", "temporalis", "orbicularis", "oculi", "risorius", "jaw", "tongue", "special"]
        # find full names in rotation parameters
        face_params = [name for name in self.joints_rotation_params.keys() if any(name.startswith(prefix) for prefix in face_prefixes)]
        return face_params
    
    def get_fingertoe_parameter_names(self):
        """
        Returns the names of the parameters used for the fingers and toes.
        """
        fingertoe_prefixes = ["toe", "finger", "metacarpal"]
        # find full names in rotation parameters
        fingertoe_params = [name for name in self.joints_rotation_params.keys() if any(name.startswith(prefix) for prefix in fingertoe_prefixes)]
        return fingertoe_params
    
    def get_bodypose_parameters(self):
        """
        Returns body pose parameters as axis-angle rotation vectors.
        """
        keys = list(self.joints_rotation_params.keys())
        values = torch.stack(list(self.joints_rotation_params.values()), dim=1)  # (bs, 162, 3)
        return keys, values
    
    def get_shape_parameters(self):
        """
        Returns shape parameters as a vector.
        """
        keys = list(self.shape_params.keys())
        values = torch.stack(list(self.shape_params.values()), dim=1)  # (bs, num_phenotypes)
        return keys, values
    
    def get_local_changes_parameters(self):
        """
        Returns local changes parameters as a vector.
        """
        keys = list(self.local_changes_kwargs.keys())
        values = torch.stack(list(self.local_changes_kwargs.values()), dim=1)
        return keys, values

    def get_parametrization(self):
        """
        Computes the current parameterization of the model:
        - Rigid transformations (as `roma.Rigid`) for each joint
        - Macrodetails (e.g., age, muscle tone, race blend)
        - Local shape refinements
        Returns:
            pose_parameters (dict): mapping from joint name to roma.Rigid
            phenotype_kwargs (dict): macro shape parameters
            local_changes_kwargs (dict): fine-level deformations
        """
        pose_parameters = dict()
        root_rotmat = roma.rotvec_to_rotmat(self.root_rotation_params)
        root_transl = self.root_translation_params.view(self.batch_size, 3)
        pose_parameters["root"] = roma.Rigid(roma.special_procrustes(root_rotmat, regularization=0.1), root_transl)
        for label, param in self.joints_rotation_params.items():
            ranges = self.joint_limit_ranges[label]
            clamped_param = clamp_but_preserve_gradients(param, ranges[0,None], ranges[1,None])
            rotmat = roma.rotvec_to_rotmat(clamped_param)
            pose_parameters[label] = roma.Rigid(rotmat, self.null_translation.expand(self.batch_size, -1))

        shape_kwargs = dict()
        for key, value in self.shape_params.items():
            shape_kwargs[key] = clamp_but_preserve_gradients(value, 0, 1)

        phenotype_kwargs = dict()
        phenotype_kwargs.update(shape_kwargs)
        
        return pose_parameters, phenotype_kwargs, self.local_changes_kwargs
    
    def get_joints(self, smplx_verts):
        """SMPLX vertices → SMPL vertices, COCO joints, dense keypoints."""
        smpl_verts = torch.einsum('ij,bjk->bik', self.smplx2smpl, smplx_verts)  # (bs, 6890, 3)
        coco_joints = vertices2joints(self.smpl2coco, smpl_verts)  # (bs, 17, 3)
        dense_kps = torch.einsum('ij,bjk->bik', self.smpl2dense, smpl_verts)  # (bs, 138, 3)
        return smpl_verts, coco_joints, dense_kps

    def forward(self):
        """
        Performs forward thought the body model.
        """

        # convert to rotmat representation
        pose_parameters, phenotype_kwargs, local_changes_kwargs = self.get_parametrization()

        # run the model
        output = self.model(pose_parameters=pose_parameters,
                            phenotype_kwargs=phenotype_kwargs,
                            local_changes_kwargs=local_changes_kwargs)
        
        v3d = output['vertices']
        
        smpl_verts, coco_joints, dense_joints = self.get_joints(v3d)  # 17 COCO + 138 dense
        
        shape = torch.stack(list(self.shape_params.values()), dim=1) # (bs, 11)
        body_pose = torch.stack(list(self.joints_rotation_params.values()), dim=1) # (bs, 162, 3)

        # scale root depth
        root_transl_flat = self.root_translation_params.view(self.batch_size, 3)
        scaled_depth = root_transl_flat[:, 2] * self.depth_params['depth_scale'] + self.depth_params['depth_shift']
        scaled_kp_depth = coco_joints[:, :, 2] * self.depth_params['depth_scale'] + self.depth_params['depth_shift']

        # print(f"depth: {self.root_translation_params[:, 2]}, scaled: {scaled_depth}, scale: {self.depth_params['depth_scale'].item()}, shift: {self.depth_params['depth_shift'].item()}")

        final_output = {'vertices': v3d,
                        'smpl_vertices': smpl_verts, # SMPL topology (6890)
                        'coco_joints': coco_joints, # coco 17 joints
                        'dense_joints': dense_joints,
                        'shape': shape,
                        'shape_dict': self.shape_params,
                        'body_pose': body_pose,
                        'root_rotation': self.root_rotation_params,
                        'root_translation': self.root_translation_params,
                        'scaled_depth': scaled_depth,
                        'scaled_kp_depth': scaled_kp_depth,
                        'depth_scale': self.depth_params['depth_scale'],
                        'depth_shift': self.depth_params['depth_shift'],
                        }

        return final_output
import os
import numpy as np
import torch
import hydra
import json
from PIL import Image
import cv2
from tqdm import tqdm
import anny
from omegaconf import DictConfig, OmegaConf

from annyfit_multiperson import AnnyfitMultiPerson
from utils import dense_kps_to_fullimage, get_focalLength_from_fieldOfView
from render import visualize_and_save


class _DummyLogger:
    """Minimal logger replacement when TensorBoard logging is disabled."""
    def __init__(self, log_dir):
        self.log_dir = log_dir


class Annyfit:
    def __init__(self, cfg: DictConfig, device='cpu'):
        self.cfg = cfg
        print("--- Configuration ---")
        print(OmegaConf.to_yaml(self.cfg))
        print("---------------------")
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        # Save the dictionary to a JSON file
        config_save_dir = os.path.join(cfg.logger.save_dir, cfg.logger.name)
        os.makedirs(config_save_dir, exist_ok=True)
        config_save_path = os.path.join(config_save_dir, "config.json")
        with open(config_save_path, "w") as f:
            json.dump(cfg_dict, f, indent=4)

        self.device = device
        self.logger_enabled = self.cfg.get('logger_enabled', True)
        self.img_prefixes = [f.split('.')[0] for f in os.listdir(cfg.data.dataset_folder)]
        anny_model = anny.create_fullbody_model(remove_unattached_vertices=False,
                                                local_changes=True, default_pose_parameterization='root_relative_world')
        self.bone_labels = anny_model.bone_labels
        self.shape_labels = anny_model.phenotype_labels

        face_prefixes = ["oris", "levator", "eye", "temporalis", "orbicularis", "oculi", "risorius", "jaw", "tongue", "special"]
        self.face_idx = torch.tensor([idx for idx, name in enumerate(self.bone_labels) if any(name.startswith(prefix) for prefix in face_prefixes)])
        fingertoe_prefixes = ["toe", "finger", "metacarpal"]
        self.fingertoe_idx = [idx for idx, name in enumerate(self.bone_labels) if any(name.startswith(prefix) for prefix in fingertoe_prefixes)]

    def _prepare_shape_attributes(self, vlm_data, shape, pids, shape_keys):
        shape_attr = {'batch_indices': [], 'attr_indices': [], 'attr_values': [], 'is_adult': []}
        for p_idx in pids:
            _p_shape_attr = vlm_data['shape_attributes'].item().get(str(p_idx), {})
            p_shape_attr = _p_shape_attr.copy()
            for key, value in _p_shape_attr.items():
                if value == -1.0:  # placeholder for invalid values
                    p_shape_attr.pop(key)
                k_idx = shape_keys.index(key) if key in shape_keys else None
                if k_idx is not None and self.cfg.init.use_vlm_shape and value != -1.0:
                    shape[p_idx][k_idx] = value
                    shape_attr['attr_indices'].append(k_idx)
                    shape_attr['attr_values'].append(value)
                    shape_attr['batch_indices'].append(p_idx)

            p_is_adult = (vlm_data['text_attributes'].item().get(str(p_idx), {}).get('age', '') == 'adult')
            shape_attr['is_adult'].append(p_is_adult)

        shape_attr['is_adult'] = torch.tensor(shape_attr['is_adult'], dtype=torch.bool)
        return shape_attr, shape

    def prepare_data(self, pids, img_data, mesh_data, vlm_data=None):
        img_path = img_data['img_path'].item()
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Camera intrinsic matrix
        if self.cfg.init.use_pred_fov and 'cam_int' in img_data:
            K = torch.tensor(img_data['cam_int'], dtype=torch.float32).unsqueeze(0)
        else:
            largest_side = max(img_width, img_height)
            fov = self.cfg.camera.fov  # in degrees
            fl = get_focalLength_from_fieldOfView(fov=fov, img_size=largest_side).squeeze()
            K = torch.tensor([[fl, 0, img_width / 2],
                              [0, fl, img_height / 2],
                              [0, 0, 1]], dtype=torch.float32).unsqueeze(0)

        all_pose = mesh_data['pose']
        shape = mesh_data['shape']
        transl = mesh_data['transl']

        shape_attr = None
        if vlm_data is not None:
            shape_attr, shape = self._prepare_shape_attributes(vlm_data, shape, pids, mesh_data['shape_keys'])

        initial_params = {
            'root_rotation_params': all_pose[:, 0, :],       # (bs, 3)
            'root_translation_params': transl.squeeze(1),    # (bs, 3)
            'body_pose_params': all_pose[:, 1:, :],          # (bs, 162, 3)
            'shape_params': shape,                            # (bs, 11)
        }

        # change the dense_kp to the full image coordinates
        dense_cropped = torch.from_numpy(img_data['dense_kps'][pids])
        bbox = torch.from_numpy(img_data['bboxes'][pids])
        dense_kp = dense_kps_to_fullimage(dense_cropped, bbox)

        kp2d = torch.from_numpy(img_data['keypoints'][pids])  # body kps in coco format (bs, 17, 3)

        masks = torch.from_numpy(img_data["masks"][pids] > 0)
        depth = torch.from_numpy(img_data['median_depth'][pids])

        target_data = {
            'keypoints_2d': kp2d,
            'dense_kp': dense_kp,
            'depth': depth,
            'masks': masks,
        }

        if shape_attr is not None:
            target_data['shape_attributes'] = shape_attr

        if 'depth_map' in img_data:
            scene_depth = torch.from_numpy(img_data['depth_map'])
            target_data['depth_map'] = scene_depth

        return initial_params, target_data, K, img_path

    def load_npz_file(self, file_path):
        try:
            loaded_npz = np.load(file_path, allow_pickle=True)
            img_data = {key: loaded_npz[key] for key in loaded_npz.files}
            return img_data
        except:
            return None

    def load_image_meshes(self, img_prefix, pids):
        mesh_folder = self.cfg.data.mesh_folder
        if 'multihmr' in mesh_folder or 'ourfits' in mesh_folder:
            data = self.load_npz_file(os.path.join(mesh_folder, f'{img_prefix}.npz'))
            if data is None:
                return None
            mesh_data = {
                'pose': torch.tensor(data['pose'][pids]),
                'transl': torch.stack([torch.as_tensor(data['multihmr_pred'][p_idx]['transl']) for p_idx in pids]),
                'shape_keys': list(data['multihmr_pred'][0]['shape_keys']),  # all person keys are in the same order
                'shape': torch.stack([torch.as_tensor(data['multihmr_pred'][p_idx]['shape']) for p_idx in pids]),
            }
        else:
            print(f"Unsupported mesh folder format: {mesh_folder}")
            return None
        return mesh_data

    def save_image_parameters(self, final_params: dict, save_dir: str, pids: list):
        save_path = os.path.join(save_dir, f"params.npz")
        final_params_cpu = {
            str(p_idx): {
                param_name: param_tensor[p_idx].cpu() if isinstance(param_tensor[p_idx], torch.Tensor) else param_tensor[p_idx]
                for param_name, param_tensor in final_params.items()
            }
            for p_idx in pids
        }
        np.savez(save_path, **final_params_cpu)

    def render_optimized_people(self, img_path: str, final_vertices: torch.Tensor, faces: torch.Tensor, camera_intrinsics: torch.Tensor, save_path: str):
        img = cv2.imread(img_path)
        visualize_and_save(image=img, vertices=final_vertices, faces=faces, K=camera_intrinsics,
                           output_path=save_path, distance=5.0, alpha=1.0, center_on_mesh=True, save_img=True)

    def optimize_image(self, img_prefix: str):
        print(f"Processing: {img_prefix}")

        if self.logger_enabled:
            image_logger = hydra.utils.instantiate(self.cfg.logger, version=img_prefix)
        else:
            log_dir = os.path.join(self.cfg.logger.save_dir, self.cfg.logger.name, f"version_{img_prefix}")
            image_logger = _DummyLogger(log_dir)

        # skip if image is already finished
        final_render_path = os.path.join(image_logger.log_dir, 'vis', f"{img_prefix}_final_allpeople.jpg")
        if os.path.exists(os.path.join(image_logger.log_dir, 'params.npz')) and os.path.exists(final_render_path):
            print(f"Skipping {img_prefix}, already processed.")
            return

        # check preprocessed data is valid
        img_info = self.load_npz_file(os.path.join(self.cfg.data.dataset_folder, f'{img_prefix}.npz'))
        if img_info is None:
            print(f'Skipping {img_prefix} invalid dataset file')
            return

        img_data = self.load_npz_file(img_info["preprocessed_path"].item())
        if img_data is None:
            print(f'Skipping {img_prefix}, invalid preprocessed file')
            return

        # load initial mesh
        mesh_data = self.load_image_meshes(img_prefix, img_data['person_ids'])
        if mesh_data is None:
            print(f'Skipping {img_prefix}, invalid mesh file')
            return

        vlm_data = self.load_npz_file(img_info["vlm_path"].item()) if "vlm_path" in img_info else None
        if vlm_data is None:
            print(f'Warning: {img_prefix}, invalid vlm file, proceeding without vlm data')

        # prepare the data for optimization
        initial_params, target_data, camera_intrinsics, img_path = self.prepare_data(
            img_data['person_ids'], img_data, mesh_data, vlm_data=vlm_data)

        # init Annyfit multi-person module
        annyfit_multiperson = AnnyfitMultiPerson(self.cfg, image_logger, initial_params,
                                                 target_data, img_path, camera_intrinsics, self.device)
        annyfit_multiperson.to(self.device)

        # run the optimization
        final_params = annyfit_multiperson.optimize()

        initial_vertices = annyfit_multiperson.get_initial_vertices()
        final_vertices = annyfit_multiperson.get_final_vertices()

        # save the parameters to a file
        self.save_image_parameters(final_params, image_logger.log_dir, img_data['person_ids'])

        faces = annyfit_multiperson.annyfit_stage.faces

        # Render the initial and final vertices of all optimized people together
        save_path = os.path.join(image_logger.log_dir, 'vis', f"{img_prefix}_initial_allpeople.jpg")
        self.render_optimized_people(img_path, initial_vertices, faces, camera_intrinsics, save_path)

        save_path = os.path.join(image_logger.log_dir, 'vis', f"{img_prefix}_final_allpeople.jpg")
        self.render_optimized_people(img_path, final_vertices, faces, camera_intrinsics, save_path)

    def run(self, valid_images=None):
        """
        Run the optimization process for all images.
        """
        if valid_images is not None:
            self.img_prefixes = [img for img in self.img_prefixes if img in valid_images]

        for img_prefix in tqdm(self.img_prefixes):
            try:
                self.optimize_image(img_prefix)
            except Exception as e:
                print(f"Error processing {img_prefix}: {e}")
                continue

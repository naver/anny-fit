import os
import glob
import tyro
import cv2 
import json
import numpy as np
import torch
import torch.nn as nn
import copy

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

from .constants import CHECKPOINTS_DIR, PREPROCESS_CONFIGS_DIR

"""
Modified from https://github.com/hongsukchoi/HSfM_RELEASE/blob/main/get_pose2d_vitpose_for_hsfm.py
"""

LEFT_HAND_IDXS = {
    'left_hand_root': 91, 'left_thumb1': 92, 'left_thumb2': 93, 'left_thumb3': 94, 'left_thumb4': 95,
    'left_forefinger1': 96, 'left_forefinger2': 97, 'left_forefinger3': 98, 'left_forefinger4': 99, 
    'left_middle_finger1': 100, 'left_middle_finger2': 101, 'left_middle_finger3': 102, 'left_middle_finger4': 103,
    'left_ring_finger1': 104, 'left_ring_finger2': 105, 'left_ring_finger3': 106, 'left_ring_finger4': 107,
    'left_pinky_finger1': 108, 'left_pinky_finger2': 109, 'left_pinky_finger3': 110, 'left_pinky_finger4': 111,
    'left_wrist': 9
}

RIGHT_HAND_IDXS = {
    'right_hand_root': 112, 'right_thumb1': 113, 'right_thumb2': 114, 'right_thumb3': 115, 'right_thumb4': 116,
    'right_forefinger1': 117, 'right_forefinger2': 118, 'right_forefinger3': 119, 'right_forefinger4': 120,
    'right_middle_finger1': 121, 'right_middle_finger2': 122, 'right_middle_finger3': 123, 'right_middle_finger4': 124,
    'right_ring_finger1': 125, 'right_ring_finger2': 126, 'right_ring_finger3': 127, 'right_ring_finger4': 128,
    'right_pinky_finger1': 129, 'right_pinky_finger2': 130, 'right_pinky_finger3': 131, 'right_pinky_finger4': 132,
    'right_wrist': 10
}

class ViTPoseModel:
    MODEL_DICT = {
        'ViTPose+-H_coco_multitask': {
            'config': os.path.join(PREPROCESS_CONFIGS_DIR, 'ViTPose_huge_wholebody_256x192.py'),
            'model':  os.path.join(CHECKPOINTS_DIR, 'vitpose_huge_wholebody.pth'),
        },
        'ViTPose+-H_interhand2d': {
            'config': os.path.join(PREPROCESS_CONFIGS_DIR, 'ViTPose_huge_interhand2d_all_256x192.py'),
            'model':  os.path.join(CHECKPOINTS_DIR, 'vitpose_huge_hand_interhand2d.pth'),
        },
        'hrnetv2_wholebody': {
            'config': os.path.join(PREPROCESS_CONFIGS_DIR, 'hrnet_w48_coco_wholebody_384x288_dark_plus.py'),
            'model':  os.path.join(CHECKPOINTS_DIR, 'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'),
        },
        'hrnetv2_hand': {
            'config': os.path.join(PREPROCESS_CONFIGS_DIR, 'hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'),
            'model':  os.path.join(CHECKPOINTS_DIR, 'hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'),
        }
    }

    def __init__(
            self, 
            model_name: str = 'ViTPose+-H (multi-task train, COCO)',
            device: str = "cuda",
            **kwargs
    ):
        
        self.device = torch.device(device)

        self.model_name = model_name

        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: List[np.ndarray],
            box_score_threshold: float = 0.5) -> List[Dict[str, np.ndarray]]:
        """
        det_results: a list of Dict[str, np.ndarray] 'bbox': xyxyc
        """
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=det_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy',)
                                        
        return out

    def predict_pose_nobboxes(
            self,
            image: np.ndarray):
        """
        det_results: a list of Dict[str, np.ndarray] 'bbox': xyxyc
        """
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               format='xyxy',)
                                        
        return out


    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: List[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis

def add_hand_keypoints(image, pose_results, hand_model, bbox_thresh):
    # for each person, get the left and right hand bboxes and run 
    offset = 25 # for bigger hand bbox
    for person in pose_results:
        keypoints_2d = person['keypoints']
        
        x_min, y_min = np.min(keypoints_2d[list(LEFT_HAND_IDXS.values()), :2], axis=0) - offset
        x_max, y_max = np.max(keypoints_2d[list(LEFT_HAND_IDXS.values()), :2], axis=0) + offset
        bbox_left = {'bbox': np.array([x_min, y_min, x_max, y_max, person['bbox'][-1]])}            
        
        x_min, y_min = np.min(keypoints_2d[list(RIGHT_HAND_IDXS.values()), :2], axis=0) - offset
        x_max, y_max = np.max(keypoints_2d[list(RIGHT_HAND_IDXS.values()), :2], axis=0) + offset
        bbox_right = {'bbox': np.array([x_min, y_min, x_max, y_max, person['bbox'][-1]])}  

        hand_results = [bbox_left, bbox_right]

        hand_pose_results = hand_model.predict_pose(image, 
                                                    hand_results, 
                                                    box_score_threshold=bbox_thresh)
    
        person['hand'] = hand_pose_results

        person['keypoints'][list(LEFT_HAND_IDXS.values())[:-1], :] = hand_pose_results[0]['keypoints']
        person['keypoints'][list(RIGHT_HAND_IDXS.values())[:-1], :] = hand_pose_results[1]['keypoints']
    return pose_results

def run_vitpose_preprocess(img_dir: str='images', 
        preprocess_file: str='preprocessed_data.npz', 
        output_dir: str='vitpose',
        model_name: str='ViTPose+-H_coco_multitask',
        vis: bool = False,
        hand_model: str = 'hrnetv2_hand'):
    """
    Run ViTPose on preprocessed images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model = ViTPoseModel(model_name=model_name, device=device)
    hand_model = ViTPoseModel(model_name=hand_model, device=device)
    # Pose estimation configuration
    box_score_threshold = 0.5
    kpt_score_threshold = 0.3
    vis_dot_radius = 4
    vis_line_thickness = 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run per image
    data = np.load(preprocess_file, allow_pickle=True)
    out_data = {}
    for img_name in tqdm(data.keys()):
        img_data = data[img_name].item()
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        # add score to bbox
        bboxes = []
        for bbox in img_data['bbox']:
            bboxes.append({'bbox': np.concatenate([bbox, np.ones(1)], dtype=np.float32)}) # bbox is xyxyc
        
        # out: List[Dict[str, np.ndarray]]; keys: bbox, keypoints. values are numpy arrays
        # num_keypoints (133, 3)
        out = model.predict_pose(image, bboxes, box_score_threshold)
        # update hands with specialized hand model
        out = add_hand_keypoints(image, out, hand_model, box_score_threshold)
        out_data[img_name] = img_data.copy()
        out_data[img_name]['gt_keypoints'] = [p_data['keypoints'][:17, :] for p_data in out]
        out_data[img_name]['all_keypoints'] = [p_data['keypoints'] for p_data in out]

        if vis:
            vis_out = model.visualize_pose_results(image, out, kpt_score_threshold,
                                            vis_dot_radius, vis_line_thickness)
            vis_out_path = os.path.join(output_dir, f'pose_{img_name}')
            cv2.imwrite(vis_out_path, vis_out)
    
    # save the output to a new file
    save_filename = preprocess_file.replace('.npz', f'_vitpose.npz')
    np.savez(save_filename, **out_data)

if __name__ == "__main__":
    tyro.cli(run_vitpose_preprocess)
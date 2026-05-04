import os
import json
import argparse
import cv2
import torch
import numpy as np
import shutil
import gc
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

from .constants import CHECKPOINTS_DIR, CAMERAHMR_DIR, MULTIHMR_DIR


def init_multihmr(model_name='multiHMR_672_L_anny', device='cuda'):
    """Load MultiHMR-Anny model from submodules/multi-hmr.

    Checkpoint must be at checkpoints/{model_name}.pt.
    Returns the loaded model ready for inference.
    """
    from multi_hmr_anny.multi_hmr import Multi_HMR as ModelAnny

    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_name + '.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"MultiHMR checkpoint not found: {ckpt_path}\n"
            f"Download with: wget -O {ckpt_path} "
            f"https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = {k: v for k, v in vars(ckpt['args']).items()}
    model = ModelAnny(**kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.eval()


def _multihmr_open_image(img_path, img_size, device):
    """Open, resize-pad, and normalize an image for MultiHMR (matches demo.open_image)."""
    from PIL import ImageOps
    from utils import normalize_rgb
    img_pil = Image.open(img_path).convert('RGB')
    img_pil_full = img_pil.copy()
    img_pil = ImageOps.contain(img_pil, (img_size, img_size))
    img_pil = ImageOps.pad(img_pil, size=(img_size, img_size))
    x = torch.from_numpy(normalize_rgb(np.asarray(img_pil))).unsqueeze(0).to(device)
    return x, img_pil_full



def _bboxes_to_patch_idx(bboxes, all_keypoints, scale, offset_x, offset_y, patch_size, n_patches, device):
    """Convert detected bboxes to patch indices for the model's score map.

    Uses ViTPose head keypoints (indices 0-4: nose/eyes/ears) when valid,
    otherwise falls back to the top-centre of the bounding box.

    Args:
        bboxes:        (N, 4) float32 [x1,y1,x2,y2] in original image coords
        all_keypoints: (N, 133, 3) float32 ViTPose whole-body keypoints, or None
        scale, offset_x, offset_y: transform from original → model img_size space
        patch_size: int, encoder patch size (e.g. 14)
        n_patches: int, number of patches per side
        device: torch device

    Returns:
        tuple(batch_t, y_t, x_t) of long tensors on device, length N
    """
    batch_indices, y_indices, x_indices = [], [], []
    for p_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        # Try to use head keypoints (COCO whole-body 0-4: nose, L/R eye, L/R ear)
        head_x, head_y = None, None
        if all_keypoints is not None:
            head_kps = all_keypoints[p_idx, :5]  # (5, 3)
            valid = head_kps[:, 2] > 0.3
            if valid.sum() > 0:
                head_x = float(head_kps[valid, 0].mean())
                head_y = float(head_kps[valid, 1].mean())

        if head_x is None:
            # Fall back: top-centre of bbox
            head_x = float((x1 + x2) / 2)
            head_y = float(y1 + (y2 - y1) * 0.15)

        # Transform to model img_size space then to patch grid
        mx = head_x * scale + offset_x
        my = head_y * scale + offset_y
        px = int(mx / patch_size)
        py = int(my / patch_size)
        px = max(0, min(n_patches - 1, px))
        py = max(0, min(n_patches - 1, py))

        batch_indices.append(0)  # single image (batch size 1)
        y_indices.append(py)
        x_indices.append(px)

    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(y_indices,     dtype=torch.long, device=device),
        torch.tensor(x_indices,     dtype=torch.long, device=device),
    )


def multihmr_preprocessed(img_list, data_root, model_name='multiHMR_672_L_anny', device='cuda'):
    """Run MultiHMR-Anny inference and save initial Anny pose/shape parameters.

    Reads cam_int and bboxes from {data_root}/preprocessed/{img}.npz.
    Saves to {data_root}/multihmr_{model_name}/{img}.npz with keys:
      - multihmr_pred: (N,) object array of per-person dicts
      - pose: (N, 163, 3) float32 rotvec

    Detections are forced to match the preprocessed bboxes via match_indices,
    so the i-th output person always corresponds to the i-th person_id.

    Returns the output folder path.
    """
    preprocessed_folder = os.path.join(data_root, 'preprocessed')
    out_folder = os.path.join(data_root, f'multihmr_{model_name}')
    os.makedirs(out_folder, exist_ok=True)

    model = init_multihmr(model_name, device=device)
    img_size = model.img_size
    patch_size = model.encoder.patch_size
    n_patches = img_size // patch_size
    shape_keys = np.array(model.body_model.shape_keys)
    bone_labels = np.array(model.body_model.bone_labels)

    for img_name in tqdm(img_list, desc='MultiHMR'):
        save_path = os.path.join(out_folder, f'{img_name[:-4]}.npz')
        if os.path.exists(save_path):
            continue

        img_data = load_npz_file(os.path.join(preprocessed_folder, f'{img_name[:-4]}.npz'))
        if img_data is None or 'cam_int' not in img_data:
            continue

        img_path = str(img_data['img_path'].item())
        cam_int    = img_data['cam_int']        # (3, 3) float32
        bboxes     = img_data['bboxes']         # (N, 4) float32
        person_ids = img_data['person_ids']     # (N,) int64
        all_kps    = img_data.get('all_keypoints', None)  # (N, 133, 3) or None
        N = len(person_ids)

        # Compute scale / padding offset: original → model img_size space
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        scale    = img_size / max(orig_w, orig_h)
        offset_x = (img_size - orig_w * scale) / 2
        offset_y = (img_size - orig_h * scale) / 2

        x, _ = _multihmr_open_image(img_path, img_size, device)

        # Build camera K in model image space from estimated cam_int
        K = torch.eye(3)
        K[0, 0] = cam_int[0, 0] * scale
        K[1, 1] = cam_int[1, 1] * scale
        K[0, 2] = cam_int[0, 2] * scale + offset_x
        K[1, 2] = cam_int[1, 2] * scale + offset_y
        K = K.unsqueeze(0).to(device)

        # Convert bboxes to patch indices; model's match_indices will snap them
        # to the nearest predicted patches (or keep them if no prediction nearby)
        bbox_idx = _bboxes_to_patch_idx(
            bboxes, all_kps, scale, offset_x, offset_y, patch_size, n_patches, device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                humans = model(x, K=K, idx=bbox_idx, is_training=False,
                               det_thresh=0.3, nms_kernel_size=3)

        # Output order is preserved: humans[i] corresponds to bboxes[i]
        pose = np.zeros((N, 163, 3), dtype=np.float32)
        multihmr_pred = np.empty(N, dtype=object)
        for local_idx, h in enumerate(humans[:N]):
            rotvec        = h['rotvec'].cpu().float().numpy()         # (163, 3)
            shape         = h['shape'].cpu().float().numpy()          # (11,)
            transl        = h['transl'].cpu().float().numpy()         # (3,)
            transl_pelvis = h['transl_pelvis'].cpu().float().numpy()  # (1, 3)
            j3d           = h['j3d'].cpu().float().numpy()            # (163, 3)
            # Convert j2d from model space to original image coordinates
            j2d = h['j2d'].cpu().float().numpy()                      # (163, 2)
            j2d[:, 0] = (j2d[:, 0] - offset_x) / scale
            j2d[:, 1] = (j2d[:, 1] - offset_y) / scale

            pose[local_idx] = rotvec
            multihmr_pred[local_idx] = {
                'rotvec': rotvec,
                'shape': shape,
                'transl': transl,
                'transl_pelvis': transl_pelvis,
                'j3d': j3d,
                'j2d': j2d.astype(np.float16),
                'fov': h['fov'].cpu().float().numpy(),
                'person_ids': np.array(person_ids[local_idx]),
                'shape_keys': shape_keys,
                'bone_labels': bone_labels,
            }

        np.savez(save_path, multihmr_pred=multihmr_pred, pose=pose)

    clear_model(model)
    print("Done running MultiHMR.")
    return out_folder


def init_cam_model(device='cpu'):
    from core.cam_model.fl_net import FLNet
    model = FLNet()
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, 'cam_model_cleaned.ckpt'), map_location=device)['state_dict']
    model.load_state_dict(ckpt)
    return model.eval().to(device)


def get_cam_intrinsics(img_cv2, cam_model, img_size=256):
    normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    img_h, img_w = img_cv2.shape[:2]
    aspect_ratio = img_w / img_h
    if aspect_ratio > 1:
        new_w, new_h = img_size, int(img_size / aspect_ratio)
    else:
        new_w, new_h = int(img_size * aspect_ratio), img_size
    resized = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    start_x, start_y = (img_size - new_w) // 2, (img_size - new_h) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    img_t = torch.from_numpy(padded.astype('float32').transpose(2, 0, 1) / 255.0)
    img_t = normalize_img(img_t).unsqueeze(0)
    device = next(cam_model.parameters()).device
    with torch.no_grad():
        estimated_fov, _ = cam_model(img_t.to(device))
    vfov = estimated_fov[0, 1]
    fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
    cam_int = np.array([[fl_h, 0, img_w / 2], [0, fl_h, img_h / 2], [0, 0, 1]], dtype=np.float32)
    return cam_int, np.float32(vfov.item())


def init_densekp(device='cuda'):
    import core.constants as _cam_consts
    _cam_consts.VITPOSE_BACKBONE = os.path.join(CHECKPOINTS_DIR, 'vitpose_backbone.pth')
    from core.densekp_trainer import DenseKP
    model = DenseKP.load_from_checkpoint(os.path.join(CHECKPOINTS_DIR, 'densekp.ckpt'), strict=False)
    return model.eval().to(device)


def predict_densekp(img_cv2, bboxes_np, kp_model, device='cuda'):
    from core.datasets.dataset import Dataset as CamDataset
    from core.utils import recursive_to
    bbox_scale = (bboxes_np[:, 2:4] - bboxes_np[:, 0:2]) / 200.0
    bbox_center = (bboxes_np[:, 2:4] + bboxes_np[:, 0:2]) / 2.0
    dataset = CamDataset(img_cv2, bbox_center, bbox_scale, None, False, '')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    dense_kps = []
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = kp_model(batch)
        dense_kps.extend([kps.cpu().detach().numpy() for kps in out['pred_keypoints']])
    return dense_kps


def _init_sam2(device='cuda'):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_image_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        os.path.join(CHECKPOINTS_DIR, 'sam2.1_hiera_large.pt'),
        device=device)
    return SAM2ImagePredictor(sam2_image_model)


def _init_groundingdino_model(device='cuda'):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, grounding_model


def init_detector(detector='groundingdino', device='cuda', **kwargs):
    """Returns a callable detect_fn(img_path) -> (bboxes_tensor, scores_tensor, PIL_image)"""
    if detector == 'groundingdino':
        processor, grounding_model = _init_groundingdino_model(device=device)
        def detect_fn(img_path):
            return detect_groundingdino(img_path, processor, grounding_model, device=device)
        return detect_fn
    elif detector == 'detectron2':
        from .detect_detectron2 import init_detectron2, run_detectron2
        from core.constants import DETECTRON_CFG
        det_model = init_detectron2(
            os.path.join(CAMERAHMR_DIR, DETECTRON_CFG),
            os.path.join(CHECKPOINTS_DIR, 'model_final_f05665.pkl'),
            threshold=kwargs.get('threshold', 0.25))
        def detect_fn(img_path):
            return run_detectron2(img_path, det_model, thresh=kwargs.get('valid_threshold', 0.4))
        return detect_fn
    else:
        raise ValueError(f"Unknown detector: {detector}")


def init_vitpose(model_name: str='ViTPose+-H_coco_multitask',
                 hand_model: str = 'hrnetv2_hand',
                 device: str = 'cuda'):
    
    from .get_2dkps_vitpose_annyfy import ViTPoseModel

    # Load the model
    model = ViTPoseModel(model_name=model_name, device=device)
    hand_model = ViTPoseModel(model_name=hand_model, device=device)
    # Pose estimation configuration
    pose_config = {
        "box_score_threshold": 0.5,
        "kpt_score_threshold": 0.3 # important for visualizing and filtering keypoints
    }
    return model, hand_model, pose_config

def init_groundingdino(device: str = "cuda"):
    """Returns (sam2_predictor, gdino_processor, gdino_model) — kept for external callers."""
    image_predictor = _init_sam2(device=device)
    processor, grounding_model = _init_groundingdino_model(device=device)
    return image_predictor, processor, grounding_model

def init_yolo(device: str = "cuda"):
    from ultralytics import YOLO

    model = YOLO(os.path.join(CHECKPOINTS_DIR, 'yolov10x.pt'))
    model.to(device)
    return model

def init_detr(device: str = "cuda"):
    from rfdetr import RFDETRLarge
    model = RFDETRLarge(pretrain_weights=os.path.join(CHECKPOINTS_DIR, 'rf-detr-large.pth'))
    return model


def init_unidepth(device: str = "cuda", model_version: str = "unidepth-v2-vitl14"):
    from unidepth.models import UniDepthV2
    # had to run: pip install -e ./submodules/UniDepth

    model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_version}")
    model = model.to(device)
    return model

def clear_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def detect_groundingdino(img_path, processor, grounding_model, text_prompt="person.", device="cuda"):
    image = Image.open(img_path)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # process the detection results
    bboxes = results[0]["boxes"]
    bbox_scores = results[0]["scores"]
    return bboxes, bbox_scores, image


def detect_detr(img_path, detr_model, person_class_index=1, det_thresh=0.3, nms_thresh=0.5):
    image = Image.open(img_path)

    detections = detr_model.predict(image, threshold=det_thresh).with_nms(threshold=nms_thresh)
    bboxes = torch.from_numpy(detections.xyxy[(detections.class_id == person_class_index)])
    bbox_scores = detections.confidence[(detections.class_id == person_class_index)]

    return bboxes, bbox_scores, image

def segment_groundingdino(image, bboxes, image_predictor):
    image_predictor.set_image(np.array(image))
        
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bboxes,
        multimask_output=False,
    )

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks

def load_npz_file(file_path):
    try:
        loaded_npz = np.load(file_path, allow_pickle=True)
    except:
        return None
    img_data = {key: loaded_npz[key] for key in loaded_npz.files}
    return img_data
        

def detect_segment_imglist(img_list, img_folder, output_folder, device="cuda",
                           max_num_people=10, detector='groundingdino',
                           estimate_camera=True, estimate_densekp=True):
    processed_folder = os.path.join(output_folder, "preprocessed")
    os.makedirs(processed_folder, exist_ok=True)

    detect_fn = init_detector(detector, device=device)
    sam2_predictor = _init_sam2(device)
    cam_model = init_cam_model(device=device) if estimate_camera else None
    kp_model = init_densekp(device=device) if estimate_densekp else None

    valid_list = []
    for img_name in tqdm(img_list):
        output_file = os.path.join(processed_folder, f'{img_name[:-4]}.npz')
        loaded = load_npz_file(output_file)
        if loaded is not None and "masks" in loaded and "img_path" in loaded:
            valid_list.append(img_name)
            continue

        img_path = os.path.abspath(os.path.join(img_folder, img_name))
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("truncated image, skipping:", img_name)
            continue

        bboxes, bbox_scores, image = detect_fn(img_path)
        num_people = len(bboxes)
        if num_people == 0 or num_people > max_num_people:
            continue

        img_width, img_height = image.size
        masks = segment_groundingdino(image, bboxes, sam2_predictor)

        bboxes_np = bboxes.cpu().numpy() if hasattr(bboxes, 'cpu') else np.array(bboxes)
        scores_np = bbox_scores.cpu().numpy() if hasattr(bbox_scores, 'cpu') else np.array(bbox_scores)
        bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, img_width - 1)
        bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, img_height - 1)

        save_dict = {
            'img_path': img_path,
            'imgname': img_name,
            'img_size': np.array([img_width, img_height]),
            'bboxes': bboxes_np,
            'scores': scores_np,
            'masks': masks,
            'num_people': np.array(num_people),
            'person_ids': np.arange(num_people, dtype=np.int64),
        }

        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if cam_model is not None:
            cam_int, fov = get_cam_intrinsics(img_cv2, cam_model)
            save_dict['cam_int'] = cam_int
            save_dict['fov'] = fov
        if kp_model is not None:
            dense_kps = predict_densekp(img_cv2, bboxes_np, kp_model, device=device)
            dense_kps = np.stack(dense_kps)  # (N, 138, 3) — col2 is log-sigma, not confidence
            # Replace log-sigma with binary confidence: 1.0 if in crop, 0.0 if outside
            in_crop = (dense_kps[:, :, 0] >= -0.5) & (dense_kps[:, :, 0] <= 0.5) & \
                      (dense_kps[:, :, 1] >= -0.5) & (dense_kps[:, :, 1] <= 0.5)
            dense_kps[:, :, 2] = in_crop.astype(np.float32)
            save_dict['dense_kps'] = dense_kps

        np.savez(output_file, **save_dict)
        valid_list.append(img_name)

    clear_model(sam2_predictor)
    if cam_model is not None:
        clear_model(cam_model)
    if kp_model is not None:
        clear_model(kp_model)

    print("Done detecting and segmenting people.")
    return valid_list

def keypoints_preprocessed(img_list, output_folder, device="cuda", vitpose_conf=0.3, high_quality=True):

    from .get_2dkps_vitpose_annyfy import add_hand_keypoints

    processed_folder = os.path.join(output_folder, "preprocessed")
    model, hand_model, pose_config = init_vitpose(device=device)

    valid_list = []
    for img_name in tqdm(img_list):
        output_file = os.path.join(processed_folder, f'{img_name[:-4]}.npz')
        img_data = load_npz_file(output_file)
        if img_data is None:
            print(f"Detection not found for {img_name}, skipping.")
            continue

        if img_data.get("keypoints") is not None:
            valid_list.append(img_name)
            continue

        img = np.asarray(Image.open(img_data["img_path"].item()).convert("RGB"))
        org_bboxes = torch.from_numpy(img_data["bboxes"])
        scores = torch.from_numpy(np.array(img_data["scores"], dtype=np.float32)).unsqueeze(1)
        bboxes = [{"bbox": np.concatenate([bbox, score])} for bbox, score in zip(org_bboxes, scores)]

        out = model.predict_pose(img, bboxes, pose_config["box_score_threshold"])
        out = add_hand_keypoints(img, out, hand_model, pose_config["box_score_threshold"])
        img_width, img_height = img_data["img_size"]

        out_clean = []
        valid_indices = []
        for p_idx, person in enumerate(out):
            keypoints = person['keypoints']
            keypoints[:, 2] = np.where(
                (keypoints[:, 0] < 0) | (keypoints[:, 1] < 0) |
                (keypoints[:, 0] >= img_width) | (keypoints[:, 1] >= img_height),
                0, keypoints[:, 2]
            )
            if high_quality and sum(keypoints[:17, 2] >= vitpose_conf) < 7:
                continue
            out_clean.append({"keypoints": keypoints})
            valid_indices.append(p_idx)

        if len(valid_indices) == 0:
            np.savez(output_file, **img_data)
            continue

        valid_list.append(img_name)
        img_data["all_keypoints"] = np.stack([p['keypoints'] for p in out_clean])
        img_data["keypoints"] = np.stack([p['keypoints'][:17] for p in out_clean])
        img_data["bboxes"] = img_data["bboxes"][valid_indices]
        img_data["scores"] = img_data["scores"][valid_indices]
        img_data["masks"] = img_data["masks"][valid_indices]
        img_data["num_people"] = np.array(len(valid_indices))
        img_data["person_ids"] = np.arange(len(valid_indices), dtype=np.int64)
        img_data["valid_indices"] = np.array(valid_indices)
        if 'dense_kps' in img_data:
            img_data["dense_kps"] = img_data["dense_kps"][valid_indices]

        np.savez(output_file, **img_data)

    clear_model(model)
    clear_model(hand_model)
    print("Done preprocessing keypoints.")
    return valid_list

def depthmaps_preprocessed(img_list, output_folder, device="cuda"):
    # TODO: add masks to get visibility here
    def keypoints_to_depth(keypoints, depth):
        num_people = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        depth_per_keypoint = torch.full((num_people, num_keypoints), -1.0, dtype=torch.float32)
        H, W = depth.shape[-2:]
        valid_mask = keypoints[..., 2] > 0
        x_coords_clipped = keypoints[..., 0].long().clamp(0, W-1)
        y_coords_clipped = keypoints[..., 1].long().clamp(0, H-1)
        all_depths = depth[y_coords_clipped, x_coords_clipped]
        depth_per_keypoint[valid_mask] = all_depths[valid_mask]
        kp_depth = [depth_per_keypoint[p].cpu() for p in range(num_people)]
        return kp_depth

    model = init_unidepth(device=device)
    
    for img_name in tqdm(img_list):
        # load existing data
        output_file = os.path.join(output_folder, "preprocessed", f'{img_name[:-4]}.npz')
        img_data = load_npz_file(output_file)
        if img_data is None:
            print(f"Detection not found for {img_name}, skipping.")
            continue

        if img_data.get("depth_map") is not None:
            # skip if already processed
            continue

        rgb = np.asarray(Image.open(img_data["img_path"].item()).convert("RGB"))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        predictions = model.infer(rgb_torch)

        # Metric Depth Estimation
        depth = predictions["depth"]
        depth = depth.squeeze().cpu()  # (H, W)

        # use the segmentation mask to get the median depth
        masks = torch.from_numpy(img_data["masks"].astype(bool))  # (num_people, H, W)
        num_people = masks.shape[0]

        expanded_depth = depth.expand_as(masks)
        masked_depth = torch.where(masks, expanded_depth, torch.tensor(float('nan')))
        masked_depth = masked_depth.view(num_people, -1)
        median_depth = torch.nanmedian(masked_depth, dim=1).values  # (num_people,)
        img_data["median_depth"] = median_depth.cpu()
        
        # # get the keypoint depth. Invalid keypoint depth is set to -1
        keypoints = torch.from_numpy(np.stack(img_data["keypoints"]))  # (num_people, num_keypoints, 3)
        img_data["keypoint_depth"] = keypoints_to_depth(keypoints, depth)
        img_data["depth_map"] = depth

        # # add dense keypoint depth if available
        # if "dense_kps" in img_data:
        #     dense_cropped = torch.from_numpy(img_data['dense_kps']) # (num_people, 138, 3)
        #     bboxes = torch.from_numpy(img_data['bboxes']) # (num_people, 4)
        #     dense_kp = torch.stack([dense_kps_to_fullimage(p_kps, bbox) for p_kps, bbox in zip(dense_cropped, bboxes)]) # (num_people, 138, 3)
        #     img_data["full_image_dense_kps"] = dense_kp
        #     img_data["dense_keypoint_depth"] = keypoints_to_depth(dense_kp, depth)

        # depth_color = colorize(depth.numpy(), vmin=0.01, vmax=10.0, cmap="magma_r")
        # artifact = image_grid([rgb, depth_color], 1,2)
        # save_path = os.path.join(output_folder, "depthmaps", f"{img_name[:-4]}.png")
        # Image.fromarray(artifact).save(save_path)
    
        # update the output file
        np.savez(output_file, **img_data)
    
    clear_model(model)
    print("Done preprocessing depthmaps.")

def vlm_attributes_preprocessed(img_list, output_folder, device="cuda", version='headcrop', batch_size=1):
    """
    Query a VLM for person attributes.
    """
    from estimate.vlm_descriptors import load_model, get_dataloader, predict_image
    from estimate.vlm_attribute_utils import attributes2shape

    if version == 'crop':
        vlm_args = {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "prompt_keys": ["age_agedef_verbose", "gender_verbose"],
                    "dataset_name": "ImageCropDetection",
                    "batch_size": batch_size,}
    elif version == 'headcrop':
        vlm_args = {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "prompt_keys": ["age_agedef_verbose", "gender_verbose"],
                    "dataset_name": "ImageHeadCropDetection",
                    "batch_size": batch_size,}
    elif version == 'maskedcrop':
        vlm_args = {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "prompt_keys": ["age_agedef_verbose", "gender_verbose"],
                    "dataset_name": "ImageMaskedCropDetection",
                    "batch_size": batch_size,}
    else:
        raise ValueError(f"Unknown VLM version: {version}. Use 'crop', 'headcrop', or 'maskedcrop'.")
    
    vlm_folder = os.path.join(output_folder, f"vlm_estimate_{version}")
    os.makedirs(vlm_folder, exist_ok=True)

    # Load model and processor
    model, processor, max_tokens, message, vision_processor = load_model(vlm_args["model_name"], device=device)
    
    for img_name in tqdm(img_list):
        out_file = os.path.join(vlm_folder, f"{img_name[:-4]}.npz")

        if os.path.exists(out_file):
            # skip if already processed
            continue
        out = {}
        # query each person in the image
        vlm_args["preprocessed_file"] = os.path.join(output_folder, "preprocessed", f"{img_name[:-4]}.npz")
        img_args = argparse.Namespace(**vlm_args)
        try:
            dataset, dataloader = get_dataloader(img_args, processor,
                                    message=message, vision_processor=vision_processor)
            out_text = predict_image(model, max_tokens, dataset, dataloader, processor, img_args.model_name, device=device)
            out["text_attributes"] = out_text
            
            # convert the text attributes to shape values
            out["shape_attributes"] = {p_idx: attributes2shape(person_pred) for p_idx, person_pred in out_text.items()}
        
            # save the output
            np.savez(out_file, **out)
        except Exception as e:
            print(f"VLM attribute estimation failed for {img_name}: {e}")
            continue
    
    clear_model(model)


def preprocess_dataset(img_list, img_folder, output_folder, min_num_people=1, max_num_people=10,
                       vlm_version='headcrop', vlm_batch_size=1,
                       detector='groundingdino', high_quality=True,
                       estimate_camera=True, estimate_densekp=True,
                       multihmr_model='multiHMR_672_L_anny'):
    """
    Detect people, estimate camera + dense keypoints, run ViTPose, MultiHMR, depth, and VLM attributes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    valid_list = detect_segment_imglist(
        img_list, img_folder, output_folder,
        device=device, max_num_people=max_num_people,
        detector=detector, estimate_camera=estimate_camera, estimate_densekp=estimate_densekp)

    valid_list = keypoints_preprocessed(
        valid_list, output_folder, device=device, high_quality=high_quality)

    multihmr_preprocessed(valid_list, output_folder, model_name=multihmr_model, device=device)

    depthmaps_preprocessed(valid_list, output_folder, device=device)

    vlm_attributes_preprocessed(valid_list, output_folder, device=device,
                                 version=vlm_version, batch_size=vlm_batch_size)
    return valid_list


def build_dataset_from_folder(data_root, dataset_name, preprocess_data=False,
                               vlm_version='headcrop', vlm_batch_size=1,
                               detector='groundingdino', high_quality=True,
                               estimate_camera=True, estimate_densekp=True,
                               multihmr_model='multiHMR_672_L_anny'):
    """
    Build a dataset from a large scale image folder
    """
    img_folder = os.path.join(data_root, "images")
    num_people = 10  # to exclude crowds
    img_list = os.listdir(img_folder)

    if preprocess_data:
        img_list = preprocess_dataset(img_list, img_folder, data_root,
                                      max_num_people=num_people,
                                      vlm_version=vlm_version,
                                      vlm_batch_size=vlm_batch_size,
                                      detector=detector,
                                      high_quality=high_quality,
                                      estimate_camera=estimate_camera,
                                      estimate_densekp=estimate_densekp,
                                      multihmr_model=multihmr_model)
    
    if dataset_name == "single_person":
        max_person_subsets(img_list, data_root, max_num_people=1, vlm_version=vlm_version)
    elif dataset_name == "multi_person":
        max_person_subsets(img_list, data_root, max_num_people=num_people, vlm_version=vlm_version)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def max_person_subsets(valid_list, data_root, max_num_people=1, vlm_version='overlay'):
    """
    For an image folder find the subset of images that match the subset criteria.
    - person age: age classes baby, toddler, child, teenage, adult, senior
    - visibility: full (all body kps), partial (more than half), low (less than half)
    - proximity: close (less than 2m), medium (2-5m), far (more than 5m)
    """
    # create a dataset file and tag with the subset criteria
    dataset_folder = os.path.join(data_root, f"person_dataset_{max_num_people}_{vlm_version}")
    os.makedirs(dataset_folder, exist_ok=True)

    for img_name in tqdm(valid_list):
        dataset_file = os.path.join(dataset_folder, f'{img_name[:-4]}.npz')
        img_info = {}
        if os.path.exists(dataset_file):
            # skip if already processed
            continue
        # load preprocessed file
        preprocessed_file = os.path.join(data_root, "preprocessed", f'{img_name[:-4]}.npz')
        img_data = load_npz_file(preprocessed_file)
        if img_data is None:
            continue

        img_info["preprocessed_path"] = os.path.abspath(preprocessed_file)

        # check that the image has only one person
        num_people = img_data.get("num_people", 0)
        if num_people == 0 or num_people > max_num_people:
            continue

        # add vlm attributes if they exist
        vlm_file = os.path.join(data_root, f"vlm_estimate_{vlm_version}", f'{img_name[:-4]}.npz')

        vlm_data = load_npz_file(vlm_file)
        if vlm_data is not None:
            vlm_data = vlm_data['text_attributes'].item()
            img_info["vlm_path"] = os.path.abspath(vlm_file)
        
        attr = get_attributes(img_data, vlm_data=vlm_data)
        # add attributes to the img_data
        img_info["attributes"] = attr

        # save the file to the dataset folder
        np.savez(dataset_file, **img_info)

    print(f"Total number of images in the dataset: {len(os.listdir(dataset_folder))}")
    
    # print dataset statistics
    print_dataset_statistics(dataset_folder)


def map_synonyms(name, original_list):
    if name in original_list:
        return name
    synonym_map = {"kid": "child", "teen": "teenager", "elder": "senior", "woman": "female", "man": "male"}
    return synonym_map.get(name, "unknown")


def get_attributes(img_data, vlm_data=None):

    # visibility
    keypoints = np.stack(img_data["keypoints"])
    num_kps = keypoints.shape[1]
    valid = keypoints[..., 2] > 0.3  # from vitpose
    num_valid_kps = valid.sum(axis=1)
    visibility = {
        "full": (num_valid_kps == num_kps).sum(),
        "partial": ((num_valid_kps > num_kps / 2) & (num_valid_kps < num_kps)).sum(),
        "low": (num_valid_kps <= num_kps / 2).sum()
    }

    # proximity
    median_depth = img_data["median_depth"]
    proximity = {
        "close": (median_depth < 2).sum(),
        "medium": ((median_depth >= 2) & (median_depth < 5)).sum(),
        "far": (median_depth >= 5).sum()
    }

    attr = {"num_people": img_data["num_people"],
        "visibility": visibility,
        "proximity": proximity}
    
    if vlm_data:
        # age and gender from VLM
        ages = ["baby", "toddler", "child", "teenager", "adult", "senior", "unknown"]
        genders = ["female", "male", "unknown"]
        age_counts = {age: 0 for age in ages}
        gender_counts = {gender: 0 for gender in genders}

        for _, vlm_attr in vlm_data.items():
            age = vlm_attr.get("age", "unknown")
            age = map_synonyms(age, ages)
            gender = vlm_attr.get("gender", "unknown")
            gender = map_synonyms(gender, genders)
            age_counts[age] += 1
            gender_counts[gender] += 1

        attr["age"] = age_counts
        attr["gender"] = gender_counts

    return attr
    
def print_dataset_statistics(dataset_folder):

    dataset_stats = {"num_people_total": 0,
                     "visible_full": 0,
                     "visible_partial": 0,
                     "visible_low": 0,
                     "proximity_close": 0,
                     "proximity_medium": 0,
                     "proximity_far": 0,
                     "num_baby": 0,
                     "num_toddler": 0,
                     "num_child": 0,
                     "num_teenager": 0,
                     "num_adult": 0,
                     "num_senior": 0,
                     "num_female": 0,
                     "num_male": 0,
                     "num_unknown": 0,
                     }
    dataset_files = os.listdir(dataset_folder)
    # use attr in img_data to calculate statistics
    for dataset_file in dataset_files:
        img_info = load_npz_file(os.path.join(dataset_folder, dataset_file))
        img_attr = img_info["attributes"].item()
        
        # number of people
        num_people = img_attr["num_people"]
        dataset_stats["num_people_total"] += num_people
        
        # visibility
        visibility = img_attr["visibility"]
        dataset_stats["visible_full"] += visibility["full"]
        dataset_stats["visible_partial"] += visibility["partial"]
        dataset_stats["visible_low"] += visibility["low"]

        # proximity
        proximity = img_attr["proximity"]
        dataset_stats["proximity_close"] += proximity["close"]
        dataset_stats["proximity_medium"] += proximity["medium"]
        dataset_stats["proximity_far"] += proximity["far"]

        # age
        if "age" in img_attr:
            age_counts = img_attr["age"]
            for age, count in age_counts.items():
                dataset_stats[f"num_{age}"] += count
                
        if "gender" in img_attr:
            gender_counts = img_attr["gender"]
            for gender, count in gender_counts.items():
                dataset_stats[f"num_{gender}"] += count
            
    dataset_stats["mean_people_per_image"] = dataset_stats["num_people_total"] / len(dataset_files)
    
    # print dataset statistics
    for key, value in dataset_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset from folder")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder of the dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["single_person", "multi_person"],
                        help="Name of the dataset to build")
    parser.add_argument("--preprocess_data", action="store_true", default=False,
                        help="Whether to preprocess the data")
    parser.add_argument("--vlm_version", type=str, default="headcrop",
                        choices=["crop", "headcrop", "maskedcrop"],
                        help="VLM crop mode")
    parser.add_argument("--vlm_batch_size", type=int, default=4,
                        help="Batch size for VLM inference")
    parser.add_argument("--detector", type=str, default="groundingdino",
                        choices=["groundingdino", "detectron2"],
                        help="Person detector backend")
    parser.add_argument("--high_quality", action="store_true", default=True,
                        help="Filter out people with not enough valid body keypoints")
    parser.add_argument("--no_camera", action="store_true", default=False,
                        help="Skip FLNet camera intrinsics estimation")
    parser.add_argument("--no_densekp", action="store_true", default=False,
                        help="Skip CameraHMR dense keypoint estimation")
    parser.add_argument("--multihmr_model", type=str, default="multiHMR_672_L_anny",
                        help="MultiHMR checkpoint name (must exist in checkpoints/)")
    args = parser.parse_args()

    build_dataset_from_folder(args.data_root, args.dataset_name,
                              preprocess_data=args.preprocess_data,
                              vlm_version=args.vlm_version,
                              vlm_batch_size=args.vlm_batch_size,
                              detector=args.detector,
                              high_quality=args.high_quality,
                              estimate_camera=not args.no_camera,
                              estimate_densekp=not args.no_densekp,
                              multihmr_model=args.multihmr_model)

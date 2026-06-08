# Only imported when --detector detectron2 is specified.
# Requires detectron2 installed and CameraHMR on PYTHONPATH (via setup.sh).
import cv2
import torch
import numpy as np
from PIL import Image
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy


def init_detectron2(cfg_path, ckpt_path, threshold=0.25):
    cfg = LazyConfig.load(str(cfg_path))
    cfg.train.init_checkpoint = ckpt_path
    for pred in cfg.model.roi_heads.box_predictors:
        pred.test_score_thresh = threshold
    return DefaultPredictor_Lazy(cfg)


def run_detectron2(img_path, detector, thresh=0.4):
    img_cv2 = cv2.imread(str(img_path))
    out = detector(img_cv2)
    inst = out['instances']
    valid = (inst.pred_classes == 0) & (inst.scores > thresh)
    bboxes = inst.pred_boxes.tensor[valid].cpu()
    scores = inst.scores[valid].cpu()
    image = Image.open(img_path).convert("RGB")
    return bboxes, scores, image

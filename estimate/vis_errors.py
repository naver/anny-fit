import os
import numpy as np
import pickle
from PIL import Image, ImageDraw
from tqdm import tqdm

relative_age_types = ['adult', 'teen', 'kid', 'baby']
AGE_MAPPING = {'baby': 0.0, 'toddler': 0.1, 'kid': 0.330, 'child': 0.330, 'teenager': 0.500, 'teen': 0.500, 'adult': 0.660, 'elder': 0.999, 'senior': 0.999, 'unknown': 0.5}

def anny2age(norm_age):
    """
    Map from Anny model to RH age
    Uses RH cat ordering from relative_age_types
    Approximate mapping from vis:
    'baby': 0.0, 'kid': 0.330, 'teen': 0.5, 'adult': 0.66
    """
    # make sure the age is in the shape
    if norm_age < 0.33:
        return 3.0 # baby
    elif norm_age < 0.5:
        return 2.0 # kid
    elif norm_age < 0.66:
        return 1.0 # teen
    else:
        return 0.0 # adult

path = "/beegfs/scratch/user/lmaria/social_3d_pose/rh_haschild/val/vlm_descriptors/Qwen2.5-VL-7B-Instruct_overlay.pkl"
ann_path = "/beegfs/scratch/project/humans/RelativeHuman/val_annots.npz"
img_folder = "/beegfs/scratch/project/humans/RelativeHuman/images" 
vis_folder = "/beegfs/scratch/user/lmaria/social_3d_pose/rh_haschild/val/vlm_descriptors/vis"

os.makedirs(vis_folder, exist_ok=True)

with open(path, 'rb') as f:
    vlm_pred = pickle.load(f)

anns = np.load(ann_path, allow_pickle=True)['annots'].item()

for img_name in tqdm(vlm_pred.keys()):

    # img_name = "100018.jpg"
    img_pred = vlm_pred[img_name]
    img_data = anns[img_name]

    # for every person in img_data overlay bbox, gt age and pred age
    img = Image.open(os.path.join(img_folder, img_name))

    for p_idx, p_ann in enumerate(img_data):
        gt_age = relative_age_types[p_ann['age']]
        pred_age = img_pred[p_idx]['age']

        num_gt_age = p_ann['age']
        num_pred_age = anny2age(AGE_MAPPING[pred_age])
        correct_age = num_gt_age == num_pred_age

        if 'bbox_wb' in p_ann:
            bbox = p_ann['bbox_wb']
        else:
            bbox = p_ann['bbox']

        # convert bbox to [(x1, y1), (x2, y2)]
        bbox_draw = [
            (int(bbox[0]), int(bbox[1])),  # (x1, y1)
            (int(bbox[2]), int(bbox[3]))   # (x2, y2)
        ]
        # Draw bounding box
        if correct_age:
            bbox_color = 'green'
        else:
            bbox_color = 'red'
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_draw, outline=bbox_color, width=2)
        
        # Draw age
        text = f"GT: {gt_age}, Pred: {pred_age}"
        draw.text((bbox[0], bbox[1] - 10), text, fill='white')

    # Save the image with overlay
    vis_path = os.path.join(vis_folder, f"{img_name}_overlay.png")
    img.save(vis_path)


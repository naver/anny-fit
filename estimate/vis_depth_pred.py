import os
import numpy as np
import pickle
from PIL import Image, ImageDraw
from tqdm import tqdm


path = "/beegfs/scratch/user/lmaria/rh_haschild/preprocessed"
ann_path = "/beegfs/scratch/project/humans/RelativeHuman/test_annots.npz"
img_folder = "/beegfs/scratch/project/humans/RelativeHuman/images" 
vis_folder = "/beegfs/scratch/user/lmaria/rh_haschild/vis/depth_pred"

# create depth colors as a gradient of blues
num_depth_layers = 15
depth_colors = {i: (0, 0, int(255 * (i / num_depth_layers))) for i in range(num_depth_layers)}

os.makedirs(vis_folder, exist_ok=True)

def load_npz_file(file_path):
    try:
        loaded_npz = np.load(file_path, allow_pickle=True)
        img_data = {key: loaded_npz[key] for key in loaded_npz.files}
        return img_data
    except:
        return None

anns = np.load(ann_path, allow_pickle=True)['annots'].item()

for img_name, ann in tqdm(anns.items()):

    pred_path = os.path.join(path, f"{img_name[:-4]}.npz")
    img_pred = load_npz_file(pred_path)
    if img_pred is None:
        print(f"Skipping {img_name}, no prediction data found.")
        continue
    
    # calculate the error when using the median depth
    img = Image.open(os.path.join(img_folder, img_name))
    pred_depths = img_pred["median_depth"]

    # draw the bboxes and depth layers on the image
    for p_idx, p_ann in enumerate(ann):

        if 'bbox_wb' in p_ann:
            bbox = p_ann['bbox_wb']
        else:
            bbox = p_ann['bbox']
        
        gt_depth_layer = p_ann['depth_id']
        pred_depth = pred_depths[p_idx]

        # convert bbox to [(x1, y1), (x2, y2)]
        bbox_draw = [
            (int(bbox[0]), int(bbox[1])),  # (x1, y1)
            (int(bbox[2]), int(bbox[3]))   # (x2, y2)
        ]
        # Draw bounding box
        bbox_color = depth_colors[gt_depth_layer]
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_draw, outline=bbox_color, width=2)
        
        # Draw age
        text = f"GT: {gt_depth_layer}, Pred: {pred_depth:.2f}"
        draw.text((bbox[0], bbox[1] - 10), text, fill='white')

    # Save the image with overlay
    vis_path = os.path.join(vis_folder, f"{img_name}_overlay.png")
    img.save(vis_path)


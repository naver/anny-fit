"""Overlay all preprocessing elements on the original image for visual QA.

Usage:
    python -m visualize.visualize_preprocessing demo/images/36039.jpg demo/preprocessed/36039.npz \
        --vlm demo/vlm_estimate_headcrop/36039.npz \
        --multihmr demo/multihmr_multiHMR_672_L_anny/36039.npz
"""
import argparse
import os
import numpy as np
import cv2
import torch
import roma
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # upper body
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

PERSON_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
]


def person_color(idx):
    return np.array(PERSON_COLORS[idx % len(PERSON_COLORS)]) / 255.0


def draw_keypoints(ax, keypoints, person_idx, conf_thr=0.3):
    color = person_color(person_idx)
    kps = keypoints[:17]
    for i, j in COCO_SKELETON:
        if kps[i, 2] > conf_thr and kps[j, 2] > conf_thr:
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]],
                    color=color, linewidth=1.5, alpha=0.8)
    valid = kps[:, 2] > conf_thr
    ax.scatter(kps[valid, 0], kps[valid, 1], c=[color], s=15, zorder=5,
               edgecolors='white', linewidths=0.5)


def dense_kps_to_fullimage(keypoints_norm, bbox_xyxy):
    """Project normalized dense keypoints ([-0.5, 0.5] crop coords) to full-image pixels.
    keypoints_norm: (N, 138, 3), bbox_xyxy: (N, 4)."""
    bbox_center = ((bbox_xyxy[:, :2] + bbox_xyxy[:, 2:]) / 2.0)[:, None, :]  # (N, 1, 2)
    bbox_wh = bbox_xyxy[:, 2:] - bbox_xyxy[:, :2]
    longest_side = bbox_wh.max(axis=-1)
    scale = (longest_side / 200.0)[:, None, None]  # (N, 1, 1)
    kps_img = keypoints_norm.copy()
    kps_img[:, :, :2] = 200.0 * keypoints_norm[:, :, :2] * scale + bbox_center
    return kps_img


def render_anny_overlay(img, multihmr_data, cam_int):
    """Render anny meshes from multihmr predictions onto image using pyrender."""
    import pyrender
    import trimesh
    import anny

    OPENCV_TO_OPENGL = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    preds = multihmr_data['multihmr_pred']
    N = len(preds)

    # Create anny body model (same config as multihmr)
    body_model = anny.create_fullbody_model(
        remove_unattached_vertices=False, all_phenotypes=True).to(dtype=torch.float32)
    body_model.set_skinning_method('lbs')
    faces_t = body_model.get_triangular_faces()
    faces_np = faces_t.cpu().numpy()
    person_center_idx = body_model.bone_labels.index('head')
    phenotype_keys = ['age', 'gender', 'weight', 'height', 'muscle', 'proportions']

    H, W = img.shape[:2]
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=np.ones(3) * 0.1)

    for p_idx in range(N):
        p = preds[p_idx]
        rotvec = torch.tensor(p['rotvec'], dtype=torch.float32).unsqueeze(0)  # (1, 163, 3)
        shape_vals = p['shape']  # (11,)
        transl = p['transl']  # (3,)

        # Convert rotvec to rotation matrices
        rotmat = roma.rotvec_to_rotmat(rotvec.reshape(-1, 3)).reshape(1, 163, 3, 3)

        # Build homogeneous rotation matrices (1, 163, 4, 4)
        zeros_col = torch.zeros(1, 163, 3, 1, dtype=torch.float32)
        top = torch.cat([rotmat, zeros_col], dim=-1)
        bottom = torch.tensor([0, 0, 0, 1], dtype=torch.float32).expand(1, 163, 1, 4)
        rotmat_homo = torch.cat([top, bottom], dim=-2)

        # Build phenotype kwargs (only the 6 body params, not race/cupsize/firmness)
        phenotype_kwargs = {}
        for l, k in enumerate(body_model.phenotype_labels):
            if k in phenotype_keys:
                phenotype_kwargs[k] = torch.tensor([shape_vals[l]], dtype=torch.float32)

        # Forward pass
        with torch.no_grad():
            output = body_model(pose_parameters=rotmat_homo, phenotype_kwargs=phenotype_kwargs)

        verts = output['vertices']  # (1, V, 3)
        j3d = output['bone_poses'][:, :, :3, -1]  # (1, 163, 3)
        person_center = j3d[:, [person_center_idx]]  # (1, 1, 3)

        # Apply translation
        transl_t = torch.tensor(transl).reshape(1, 1, 3)
        verts = verts - person_center + transl_t
        verts_np = verts[0].cpu().numpy()

        color = person_color(p_idx)
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=(*color, 0.7))
        mesh = trimesh.Trimesh(verts_np, faces_np)
        mesh.apply_transform(OPENCV_TO_OPENGL)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))

    fx, fy = cam_int[0, 0], cam_int[1, 1]
    cx, cy = cam_int[0, 2], cam_int[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    scene.add(camera, pose=np.eye(4))

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    import trimesh as tr
    scene.add(light, pose=tr.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]))

    color_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    foreground = color_img[:, :, :3] / 255.0
    alpha = color_img[:, :, 3:] / 255.0
    overlay = (foreground * alpha + (img / 255.0) * (1 - alpha))
    return (overlay * 255).astype(np.uint8)


def visualize_one(image_path, preprocessed_path, vlm_path=None, multihmr_path=None, out_path=None):
    """Generate a multi-panel visualization for a single preprocessed image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f'  Could not read {image_path}, skipping.')
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.load(preprocessed_path, allow_pickle=True)

    num_people = int(data['num_people'])
    keypoints = data['all_keypoints']  # (N, 133, 3)
    bboxes = data['bboxes']  # (N, 4)
    masks = data['masks']  # (N, H, W)
    depth_map = data['depth_map']  # (H, W)
    dense_kps = data['dense_kps']  # (N, 138, 3) normalized
    fov = float(data['fov'])
    cam_int = data['cam_int']
    img_size = data['img_size']  # (W, H) or (H, W)

    vlm_attrs = None
    if vlm_path and os.path.isfile(vlm_path):
        vlm_data = np.load(vlm_path, allow_pickle=True)
        vlm_attrs = vlm_data['text_attributes'].item()

    has_multihmr = multihmr_path and os.path.isfile(multihmr_path)
    ncols = 4 if has_multihmr else 3
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 10))
    H, W = img.shape[:2]

    # --- Panel 1: Detections + Keypoints + Dense Keypoints + VLM labels ---
    ax = axes[0]
    ax.imshow(img)
    ax.set_title('Detections + Keypoints + VLM', fontsize=11)
    dense_kps_full = dense_kps_to_fullimage(dense_kps, bboxes)
    for p in range(num_people):
        color = person_color(p)
        x1, y1, x2, y2 = bboxes[p]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        draw_keypoints(ax, keypoints[p], p)
        # Dense keypoints
        kps = dense_kps_full[p]
        valid = kps[:, 2] > 0
        ax.scatter(kps[valid, 0], kps[valid, 1], c=[color], s=4, alpha=0.4,
                   edgecolors='none', zorder=4)
        label = f'P{p}'
        if vlm_attrs and str(p) in vlm_attrs:
            a = vlm_attrs[str(p)]
            label += f" {a.get('age', '?')} {a.get('gender', '?')}"
        ax.text(x1, y1 - 5, label, color=color, fontsize=10,
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    # --- Panel 2: Masks overlay ---
    ax = axes[1]
    ax.imshow(img)
    ax.set_title('Segmentation Masks', fontsize=11)
    mask_overlay = np.zeros((*masks.shape[1:], 4))
    for p in range(num_people):
        color = person_color(p)
        m = masks[p]
        mask_overlay[m > 0.5] = [*color, 0.4]
    ax.imshow(mask_overlay)
    ax.axis('off')

    # --- Panel 3: Depth map ---
    ax = axes[2]
    ax.set_title('Depth Map (UniDepth)', fontsize=11)
    depth_vis = depth_map.copy()
    depth_vis[depth_vis <= 0] = np.nan
    im = ax.imshow(depth_vis, cmap='magma_r')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='depth (m)')
    ax.axis('off')

    # --- Panel 4: MultiHMR mesh render ---
    if has_multihmr:
        ax = axes[3]
        multihmr_data = np.load(multihmr_path, allow_pickle=True)
        rendered = render_anny_overlay(img, multihmr_data, cam_int)
        ax.imshow(rendered)
        ax.set_title('MultiHMR Meshes (Anny)', fontsize=11)
        ax.axis('off')

    # Camera intrinsics text
    fx, fy = cam_int[0, 0], cam_int[1, 1]
    cx, cy = cam_int[0, 2], cam_int[1, 2]
    fig.suptitle(
        f'{os.path.basename(image_path)} — {num_people} people\n'
        f'Camera: fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}  '
        f'fov={np.degrees(fov):.1f}deg  img={img_size[0]}x{img_size[1]}',
        fontsize=13)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_path}')
    return fig


def find_subdir_file(data_root, prefix, pattern):
    """Find {prefix}.npz in first subdirectory matching pattern."""
    for d in sorted(os.listdir(data_root)):
        if d.startswith(pattern):
            p = os.path.join(data_root, d, f'{prefix}.npz')
            if os.path.isfile(p):
                return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize preprocessing results. '
                    'Single-image mode: pass image + preprocessed paths. '
                    'Folder mode: pass --data_root to auto-discover files.')
    # Single-image mode
    parser.add_argument('image', nargs='?', default=None, help='Path to original image')
    parser.add_argument('preprocessed', nargs='?', default=None, help='Path to preprocessed .npz')
    parser.add_argument('--vlm', default=None, help='Path to VLM estimate .npz')
    parser.add_argument('--multihmr', default=None, help='Path to multihmr .npz')
    # Folder mode
    parser.add_argument('--data_root', default=None,
                        help='Root folder containing images/, preprocessed/, etc.')
    parser.add_argument('-n', type=int, default=10, help='Number of images to sample (folder mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    # Common
    parser.add_argument('--out', default=None, help='Output path (single) or output dir (folder)')
    args = parser.parse_args()

    import random

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'visualize', 'output')

    if args.data_root:
        # --- Folder mode ---
        preprocessed_dir = os.path.join(args.data_root, 'preprocessed')
        if not os.path.isdir(preprocessed_dir):
            print(f'No preprocessed/ folder found in {args.data_root}')
            return

        all_npz = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.npz')])
        print(f'Found {len(all_npz)} preprocessed files in {preprocessed_dir}')
        if not all_npz:
            return

        random.seed(args.seed)
        sample = random.sample(all_npz, min(args.n, len(all_npz)))
        out_dir = args.out or out_dir
        os.makedirs(out_dir, exist_ok=True)

        # find original image extension
        img_dir = os.path.join(args.data_root, 'images')
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)} if os.path.isdir(img_dir) else {}

        pbar = tqdm(sample)
        for npz_name in pbar:
            prefix = os.path.splitext(npz_name)[0]
            pbar.set_description(prefix)
            img_name = img_files.get(prefix)
            if img_name is None:
                tqdm.write(f'  No image found for {prefix}, skipping.')
                continue
            img_path = os.path.join(args.data_root, 'images', img_name)
            preprocessed = os.path.join(preprocessed_dir, npz_name)
            vlm = find_subdir_file(args.data_root, prefix, 'vlm_estimate')
            multihmr = find_subdir_file(args.data_root, prefix, 'multihmr')
            out_path = os.path.join(out_dir, f'{prefix}_preview.png')
            visualize_one(img_path, preprocessed, vlm, multihmr, out_path)
        print(f'Done. Output in {out_dir}')

    elif args.image and args.preprocessed:
        # --- Single-image mode ---
        out_path = args.out or os.path.join(
            out_dir, os.path.splitext(os.path.basename(args.image))[0] + '_preview.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        visualize_one(args.image, args.preprocessed, args.vlm, args.multihmr, out_path)

    else:
        parser.print_help()
        print('\nExamples:')
        print('  # Single image:')
        print('  python -m visualize.visualize_preprocessing demo/images/36039.jpg demo/preprocessed/36039.npz')
        print('  # Folder (sample 10):')
        print('  python -m visualize.visualize_preprocessing --data_root demo -n 10')


if __name__ == '__main__':
    main()

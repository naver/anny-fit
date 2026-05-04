#!/usr/bin/env bash
# Downloads publicly available checkpoints into checkpoints/.
# Manual downloads (CameraHMR, ViTPose, SMPL) are listed in INSTALL.md.
set -e

mkdir -p checkpoints

download() {
    local path="$1" url="$2"
    if [[ -f "$path" ]]; then
        echo "$path already exists, skipping."
    else
        echo "Downloading $path ..."
        wget -q --show-progress -O "$path" "$url"
    fi
}

download checkpoints/multiHMR_672_L_anny.pt \
    https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L_anny.pt

download checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

download checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    https://download.openmmlab.com/mmpose/wholebody/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth

download checkpoints/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth \
    https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth

# Copy joint regressors from CameraHMR if available
CAMHMR_UTILS="submodules/CameraHMR/CamSMPLify/data/train-eval-utils"
REGRESSOR_DIR="checkpoints/body_models/joint_regressors"
mkdir -p "$REGRESSOR_DIR"
for f in smplx2smpl.pkl downsample_mat.pkl; do
    if [[ -f "$REGRESSOR_DIR/$f" ]]; then
        echo "$REGRESSOR_DIR/$f already exists, skipping."
    elif [[ -f "$CAMHMR_UTILS/$f" ]]; then
        echo "Copying $f from CameraHMR ..."
        cp "$CAMHMR_UTILS/$f" "$REGRESSOR_DIR/$f"
    else
        echo "WARNING: $REGRESSOR_DIR/$f not found. Copy from CameraHMR/CamSMPLify/data/train-eval-utils/"
    fi
done

echo ""
echo "Auto-downloads complete."
echo "Manual downloads still needed (see INSTALL.md):"
echo "  - checkpoints/vitpose_huge_wholebody.pth (ViTPose)"
echo "  - checkpoints/vitpose_backbone.pth (CameraHMR)"
echo "  - checkpoints/cam_model_cleaned.ckpt (CameraHMR)"
echo "  - checkpoints/densekp.ckpt (CameraHMR)"
echo "  - checkpoints/camerahmr_checkpoint_cleaned.ckpt (CameraHMR)"
echo "  - checkpoints/model_final_f05665.pkl (CameraHMR)"
echo "  - checkpoints/body_models/SMPL_NEUTRAL.pkl (SMPL)"
echo "  - checkpoints/body_models/joint_regressors/smplx2smpl.pkl (CameraHMR)"
echo "  - checkpoints/body_models/joint_regressors/downsample_mat.pkl (CameraHMR)"

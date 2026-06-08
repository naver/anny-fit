# Installation

## Setup

```bash
git clone --recurse-submodules <repo>
cd anny-fit
bash scripts/install.sh
bash scripts/download_checkpoints.sh
source setup.sh
```

`scripts/install.sh` creates a conda env `annyfit` with Python 3.10 and installs all pip dependencies.

`scripts/download_checkpoints.sh` downloads the publicly available checkpoints into `checkpoints/`.

## Manual checkpoints

The following require registration and must be downloaded manually:

**[CameraHMR](https://camerahmr.is.tue.mpg.de/)** (free registration) — download and place in `checkpoints/`:
- `cam_model_cleaned.ckpt`, `camerahmr_checkpoint_cleaned.ckpt`, `densekp.ckpt`, `model_final_f05665.pkl`
- From `train-eval-utils.zip`: `vitpose_backbone.pth` → `checkpoints/`
- From `train-eval-utils.zip`: `J_regressor_coco_hip_smpl.npy`, `smplx2smpl.pkl`, `downsample_mat.pkl` → `checkpoints/body_models/joint_regressors/`

**[ViTPose](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR)** — download and extract with `python submodules/ViTPose/tools/model_split.py --source <file>` → `checkpoints/`

**[SMPL](https://smpl.is.tue.mpg.de/)** (free registration) — `SMPL_NEUTRAL.pkl` → `checkpoints/body_models/`

## Troubleshooting

**mmcv build fails** -- Needs `--no-build-isolation` (already set in `install.sh`). Requires a C compiler (gcc-toolset-9 or similar).

**UniDepth torch.jit error** -- The install script auto-patches this. If you see `int | tuple` JIT errors, re-run the patch from `install.sh`.

**albumentations build fails** -- Needs `--no-build-isolation` (already set in `install.sh`). Caused by `pkg_resources` removal in recent setuptools.

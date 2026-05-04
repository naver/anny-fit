import os

_PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT        = os.path.dirname(_PREPROCESS_DIR)

CHECKPOINTS_DIR      = os.path.join(REPO_ROOT, 'checkpoints')
CAMERAHMR_DIR        = os.path.join(REPO_ROOT, 'submodules', 'CameraHMR')
MULTIHMR_DIR         = os.path.join(REPO_ROOT, 'submodules', 'multi-hmr')
PREPROCESS_CONFIGS_DIR = os.path.join(_PREPROCESS_DIR, 'preprocess_configs')

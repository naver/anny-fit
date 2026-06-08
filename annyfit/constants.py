import os as _os
_PROJ_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SMPL_MODEL_DIR = _os.path.join(_PROJ_ROOT, 'checkpoints', 'body_models')
SMPL2COCO_REGRESSOR = _os.path.join(_PROJ_ROOT, 'checkpoints', 'body_models', 'joint_regressors', 'J_regressor_coco_hip_smpl.npy')
SMPLX2SMPL_REGRESSOR = _os.path.join(_PROJ_ROOT, 'checkpoints', 'body_models', 'joint_regressors', 'smplx2smpl.pkl')
SMPL2DENSE_REGRESSOR = _os.path.join(_PROJ_ROOT, 'checkpoints', 'body_models', 'joint_regressors', 'downsample_mat.pkl')
import anny
import pyrender
import trimesh
import torch

os.environ['PYOPENGL_PLATFORM'] = 'egl'


model = anny.create_fullbody_model(remove_unattached_vertices=False,
                                                with_eyes=True,
                                                with_tongue=True,
                                                with_local_changes=True,
                                                default_pose_parameterization='root_relative_world',
                                                topology='smplx',
                                                ).to(dtype=torch.float32)

# pose the mesh in canonical pose

# save the mesh with the dense keypoints overlaid
dense_kp =

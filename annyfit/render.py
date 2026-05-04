import os
import torch
import math
import cv2
import numpy as np
import trimesh
import pyrender
import itertools


os.environ['PYOPENGL_PLATFORM'] = 'egl'

# transformation matrix from OpenCV to OpenGL coordinate conventions
OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def get_colors():
    colors = {'light_purple': np.array([175, 141, 195]), 'blue': np.array([67, 171, 203])}
    return colors

def visualize_and_save(image, vertices, faces, K, output_path, side_view_angle_deg=90, distance=10.0, alpha=0.9, center_on_mesh=False, save_img=False):
    """
    Renders the mesh overlaid on an image and a side-view, then saves the result.
    
    Args:
        ...
        center_on_mesh (bool): If True, the side view camera will be centered on the mesh. 
                               If False, it will be at a fixed world coordinate.
    """
    img_height, img_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    vertices = vertices.detach().cpu().numpy()
    if vertices.ndim == 2: vertices = np.expand_dims(vertices, 0)
    faces = faces.detach().cpu().numpy().squeeze()
    K = K.detach().cpu().numpy().squeeze()

    mesh_color = get_colors()['blue']
    overlay_material = pyrender.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., alpha)
    )
 
    overlay_camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    overlay_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)

    # Render Overlay View
    overlay_renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    scene_overlay = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=np.ones(3) * 0.1)
    batch_size = vertices.shape[0]
    for i in range(batch_size):
        person_vertices = vertices[i]
        mesh_overlay = trimesh.Trimesh(person_vertices, faces)
        mesh_overlay.apply_transform(OPENCV_TO_OPENGL_CAMERA_CONVENTION)
        # # save mesh to file
        # stage_name = os.path.basename(output_path)[:-4]
        # mesh_save_path = os.path.join(os.path.dirname(output_path), "meshes", f"{stage_name}_{i}.ply")
        # os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
        # mesh_overlay.export(mesh_save_path)
        scene_overlay.add(pyrender.Mesh.from_trimesh(mesh_overlay, material=overlay_material))
    
    scene_overlay.add(overlay_camera, pose=np.eye(4))
    scene_overlay.add(overlay_light, pose=trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]))
    
    color, _ = overlay_renderer.render(scene_overlay, flags=pyrender.RenderFlags.RGBA)
    foreground = color[:, :, :3] / 255.0
    alpha_channel = color[:, :, 3:] / 255.0
    background = image_rgb / 255.0
    overlay_img = foreground * alpha_channel + background * (1 - alpha_channel)
    overlay_renderer.delete()

    # --- Render Side View ---
    side_renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    scene_side = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=np.ones(3) * 0.4)
    
    side_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., 1.0)
    )

    all_transformed_vertices = []
    for i in range(batch_size):
        person_vertices = vertices[i]
        mesh_side = trimesh.Trimesh(person_vertices, faces)
        mesh_side.apply_transform(OPENCV_TO_OPENGL_CAMERA_CONVENTION)
        scene_side.add(pyrender.Mesh.from_trimesh(mesh_side, material=side_material))
        all_transformed_vertices.append(mesh_side.vertices)
    
    if center_on_mesh and all_transformed_vertices:
        combined_vertices = np.concatenate(all_transformed_vertices, axis=0)
        orbit_center = combined_vertices.mean(axis=0)
    else:
        orbit_center = np.array([0.0, 0.0, 0.0])

    # This camera automatically adjusts its clipping planes to fit the scene
    side_camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=img_width/img_height)

    target = orbit_center
    angle_rad = np.radians(side_view_angle_deg)
    eye = np.array([
        target[0] + distance * np.sin(angle_rad),
        target[1], # Camera is at the same height as the center
        target[2] + distance * np.cos(angle_rad)
    ])

    up_vector = np.array([0.0, 1.0, 0.0])

    forward_vector = target - eye
    forward_vector /= np.linalg.norm(forward_vector)
    right_vector = np.cross(forward_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    camera_up_vector = np.cross(right_vector, forward_vector)
    
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = camera_up_vector
    camera_pose[:3, 2] = -forward_vector
    camera_pose[:3, 3] = eye
    
    scene_side.add(side_camera, pose=camera_pose)
    
    side_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = trimesh.transformations.rotation_matrix(np.radians(-60), [1, 0, 0])
    scene_side.add(side_light, pose=light_pose)

    side_color, _ = side_renderer.render(scene_side)
    side_img = side_color.astype(np.float32) / 255.0
    side_renderer.delete()
    
    # --- Combine and Save ---
    side_img_resized = cv2.resize(side_img, (overlay_img.shape[1], overlay_img.shape[0]))
    combined_img = np.concatenate((overlay_img, side_img_resized), axis=1)    
    
    combined_img_bgr = cv2.cvtColor((combined_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if save_img:
        # concatenate the original image with the overlay and side view
        combined_img_bgr = np.concatenate((image, combined_img_bgr), axis=1)
    
    cv2.imwrite(output_path, combined_img_bgr)

def visualize_points(image, points, output_path, color=(0, 255, 0), overlay_num=False):
    """
    Visualizes 2D points on an image and saves the result.
    
    Args:
        image (np.ndarray): The input image.
        points (np.ndarray): 2D points to visualize, shape (B, N, 2).
        output_path (str): Path to save the output image.
    """
    img_with_points = image.copy()
    points = np.asarray(points)
    for batch_points in points:
        for i, point in enumerate(batch_points):
            cv2.circle(img_with_points, tuple(point.astype(int)), radius=3, color=color, thickness=-1)
            if overlay_num:
                # add number of point
                cv2.putText(img_with_points, str(i+1), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    cv2.imwrite(output_path, img_with_points)
    return img_with_points
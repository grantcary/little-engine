import math
import time

import numpy as np
from PIL import Image
from numba import cuda, njit

def rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))

def camera_rays(w, h, cam):
    aspect_ratio = w / h
    angle = math.tan(math.radians(cam.fov / 2))
    camera_distance = 1 / angle

    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    normalized_x = np.interp(x_indices, (0, w-1), (-1, 1))
    normalized_y = np.interp(y_indices, (0, h-1), (1, -1))
    
    xx = normalized_x * angle * aspect_ratio
    yy = normalized_y * angle
    cc = np.full_like(xx, camera_distance)

    a = np.stack([xx, yy, cc], axis=-1)
    ray_vectors = a / np.linalg.norm(a, axis=-1, keepdims=True)

    rotation = rotation_matrix(cam.rotation)
    rotated_ray_vectors = np.dot(ray_vectors.reshape(-1, 3), rotation.T)

    return rotated_ray_vectors.reshape(-1, 3)

@cuda.jit
def sim_rays(w, h, cam_position, rays, triangle_data, normal_data):
    for i in range(w):
        for j in range(h):
            for obj in triangle_data:
                for triangle in obj:
                    print(triangle)


def render(w, h, cam, objects, lights):
    print(cuda.gpus)
    cam_rays = camera_rays(w, h, cam)
    vertex_data = [obj.vertices.astype(np.float32) for obj in objects]
    face_vertex_data = [vertex_data[i][obj.faces] for i, obj in enumerate(objects)]
    normal_data = [obj.normals.astype(np.float32) for obj in objects]

    threads_per_block = 128
    blocks = (cam_rays.shape[0] + threads_per_block - 1) // threads_per_block

    sim_rays[blocks, threads_per_block](w, h, cam.position, cam_rays, face_vertex_data, normal_data)
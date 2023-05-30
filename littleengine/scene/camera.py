import math
import numpy as np

from littleengine.utils import rotation_matrix

class Camera:
    def __init__(self, position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0], fov=90, near=0.1, far=1000, aspect_ratio=1):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect_ratio = aspect_ratio

    def translate(self, x, y, z):
        self.position += np.array([x, y, z])

    def primary_rays(self, w, h):
        aspect_ratio = w / h
        angle = math.tan(math.radians(self.fov / 2))
        camera_distance = 1 / angle

        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
        normalized_x = np.interp(x_indices, (0, w-1), (-1, 1))
        normalized_y = np.interp(y_indices, (0, h-1), (1, -1))
        
        xx = normalized_x * angle * aspect_ratio
        yy = normalized_y * angle
        cc = np.full_like(xx, camera_distance)

        a = np.stack([xx, yy, cc], axis=-1)
        ray_vectors = a / np.linalg.norm(a, axis=-1, keepdims=True)

        rotation = rotation_matrix(self.rotation)
        rotated_ray_vectors = np.dot(ray_vectors.reshape(-1, 3), rotation.T)

        return rotated_ray_vectors.reshape(-1, 3)
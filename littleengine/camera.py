import numpy as np

class Camera:
    def __init__(self, fov=90, near=0.1, far=1000, aspect_ratio=1):
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect_ratio = aspect_ratio

    def translate(self, x, y, z):
        self.position += np.array([x, y, z])
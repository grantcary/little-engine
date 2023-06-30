import math
import numpy as np
from PIL import Image

class Skybox():
    def __init__(self, path):
        self.texture = np.asarray(Image.open(path))

    def get_texture(self, rays):
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        theta, phi = np.arctan2(rays[:, 0], rays[:, 1]), np.arccos(rays[:, 2])
        u, v = (theta + math.pi) / (2 * math.pi), phi / math.pi

        u_scaled = (u * self.texture.shape[1]).astype(np.int64)
        v_scaled = (v * self.texture.shape[0]).astype(np.int64)

        return self.texture[v_scaled, u_scaled]
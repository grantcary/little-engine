import math
import numpy as np
from PIL import Image

class Skybox():
    def __init__(self, path):
        self.texture = np.asarray(Image.open(path))

    def get_pixel(self, direction):
        theta = math.atan2(direction[0], direction[1])
        phi = math.acos(direction[2])

        u = (theta + math.pi) / (2 * math.pi)
        v = phi / math.pi

        u_scaled = int(round(u * self.texture.shape[1]))
        v_scaled = int(round(v * self.texture.shape[0]))

        return self.texture[v_scaled, u_scaled]
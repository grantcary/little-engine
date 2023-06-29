import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from littleengine import Camera, Skybox
import tools

h, w = 400, 400
image = np.zeros((h * w, 3), dtype=np.uint8)
cam = Camera(position=[0, 0, 5], rotation=[90, 0, 180], fov=90, aspect_ratio=1)
origin, rays = cam.position, cam.primary_rays(w, h)
norms = np.linalg.norm(rays, axis=-1, keepdims=True)
normalized_rays = rays / norms

skybox = Skybox('../littleengine/textures/puresky.png')

for i in range(rays.shape[0]):
    image[i] = skybox.get_pixel(normalized_rays[i])

image = Image.fromarray(image.reshape(h, w, 3), 'RGB')
image.show()
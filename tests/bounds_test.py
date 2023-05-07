import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np
from PIL import Image

import littleengine.object as object
import littleengine.camera as camera
import littleengine.render as render
import littleengine.bvh as bvh
import tools
SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

suzie = object.Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
# suzie.translate(-2, 0, 0)

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 3])
cam.rotation = np.array([0, 180, 0])

w, h = 100, 100
cam_rays = render.camera_rays(w, h, cam)

bb = bvh.Bounding_Box(suzie.vertices)

# hit1 = np.array([])
# for ray_direction in cam_rays:
#     t = bb.intersect(cam.position, ray_direction)
#     if t != None:
#         phit = cam.position + ray_direction * t
#         hit1 = np.append(hit1, phit)

img = np.full((h, w), 127 , dtype=np.uint8)
for i, ray_direction in enumerate(cam_rays):
    row, col = i // w, i % w
    img[row, col] = 255 if bb.intersect(cam.position, ray_direction) != None else 127

rendered_image = Image.fromarray(img, 'L')
rendered_image.show()
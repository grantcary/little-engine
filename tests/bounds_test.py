import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

import littleengine.object as object
import littleengine.camera as camera
import littleengine.render as render
import littleengine.bvh as bvh

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

suzie = object.Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
# suzie.translate(-2, 0, 0)

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 2])
cam.rotation = np.array([0, 180, 0])

w, h = 10, 10
cam_rays = render.camera_rays(w, h, cam)
# ray_direction = cam_rays[(w * h) // 2]

bounds = bvh.bounds_generator(suzie.vertices)
print(bounds)
bb = bvh.Bounding_Box(bounds)

for ray_directions in cam_rays:
    t = bb.intersect(cam.position, ray_directions)
    if t != None:
        phit = cam.position + ray_directions * t
        print(phit)

# img = np.full((h, w), 0 , dtype=np.uint8)
# for i, ray_direction in enumerate(cam_rays):
#     w, h = i // 10, i % 10
#     img[w, h] = 255 if bb.intersect(cam.position, ray_direction) != None else 0

# rendered_image = Image.fromarray(img, 'L')

# rendered_image.show()
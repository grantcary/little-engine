import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np
from PIL import Image

import littleengine.object as object
import littleengine.camera as camera
import littleengine.render2 as render2
from littleengine.bvh import BVH, Bounding_Box
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

objects = []
suzie = object.Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
suzie.translate(-2, 0, 0)
objects.append(suzie)

cube = object.Object('Cube', CUBE)
cube.material_type = 'diffuse'
cube.translate(2, 0, 0)
cube.color = np.array([0, 255, 0])
objects.append(cube)

lights = []
ico = object.Object('Light', ICOSPHERE)
ico.material_type = 'emissive'
ico.translate(0, 3, 3)
lights.append(ico)


w, h = 100, 100
cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 5])
cam.rotation = np.array([0, 180, 0])
primary_rays = cam.primary_rays(w, h)

# render2.render(100, 100, cam, objects, lights)
# h = bvh.bounding_volume_hierarchy(objects)

s = BVH(Bounding_Box(suzie.vertices), 0)
c = BVH(Bounding_Box(cube.vertices), 1)

# TODO: turn this into function
e = np.full((2, 3, 2), 0, dtype=float)
e[0] = s.bounding_box.bounds
e[1] = c.bounding_box.bounds
group_bounds = np.column_stack((e[:, :, 0].max(axis=0), e[:, :, 1].min(axis=0)))

scene = BVH(Bounding_Box(bounds=group_bounds), left=s , right=c)
render2.render(w, h, cam, scene, objects, lights)



# img = np.full((w, h), 0, dtype=np.uint8)
# st = time.time()
# for i, ray in enumerate(primary_rays):
#     row, col = i // w, i % w
#     search = scene.search_collision(cam.position, ray)
#     # print(search)
#     img[row, col] = 255 if search != None else 127
# print(time.time() - st)

# rendered_image = Image.fromarray(img, 'L')
# rendered_image.show()
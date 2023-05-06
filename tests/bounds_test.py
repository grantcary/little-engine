import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import littleengine.object as object
import littleengine.camera as camera
import littleengine.render as render
import littleengine.bvh as bvh

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

objects = []
suzie = object.Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
# suzie.translate(-2, 0, 0)
objects.append(suzie)

# cube = object.Object('Cube', CUBE)
# cube.material_type = 'diffuse'
# cube.translate(2, 0, 0)
# cube.color = np.array([0, 255, 0])
# objects.append(cube)

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 10])
cam.rotation = np.array([0, 180, 0])

w, h = 10, 10
cam_rays = render.camera_rays(w, h, cam)
ray = cam_rays[(w * h) / 2]

xyz_mm = bvh.bounding_box_generator(suzie.vertices)
bb = bvh.Bounding_Box(xyz_mm)
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

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 3])
cam.rotation = np.array([0, 180, 0])

w, h = 100, 100
cam_rays = cam.primary_rays(w, h)

h = bvh.bvh_generator(objects)
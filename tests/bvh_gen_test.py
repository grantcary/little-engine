import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np
from PIL import Image

import littleengine.object as object
import littleengine.camera as camera
import littleengine.render2 as render2
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

lights = []
ico = object.Object('Light', ICOSPHERE)
ico.material_type = 'emissive'
ico.translate(0, 3, 3)
lights.append(ico)

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 3])
cam.rotation = np.array([0, 180, 0])

render2.render(100, 100, cam, objects, lights)
# h = bvh.bounding_volume_hierarchy(objects)
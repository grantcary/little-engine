import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from littleengine import Object, bounding_volume_hierarchy, meshlet_gen
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'
PLANE = '../test_objects/plane.obj'

USE_BVH = False

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], rotate=[90, 0, 0], color=[255, 0, 0], reflectivity=0.01, bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], reflectivity=0.01, ior=1.3, bvh=USE_BVH)
plane = Object('Plane', PLANE, position=[0, 0, -0.3], scale=7, rotate=[100, 0, 0], color=[0, 0, 0], reflectivity=1, bvh=USE_BVH)
objects = [suzie, cube, plane]

icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 0, 0], ior=1.3, reflectivity=1, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[2, 0, 0], scale=0.5, color=[0, 0, 0], reflectivity=1, ior=1.3, bvh=USE_BVH)

st = time.time()
meshlet = meshlet_gen(teapot)
bvh = bounding_volume_hierarchy(teapot, meshlet)
print(time.time() - st)
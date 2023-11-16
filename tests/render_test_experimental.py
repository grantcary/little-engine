import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from littleengine import Camera, Object, Light, SceenParams, render_experimental, Skybox
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'
PLANE = '../test_objects/plane.obj'

USE_BVH = False

# suzie = Object('Monkey', SUZIE, position=[2, 0, 0], rotate=[90, 0, 0], color=[255, 0, 0], bvh=USE_BVH)
# cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], reflectivity=1, ior=1.3, alpha=0.5, bvh=USE_BVH)
suzie = Object('Monkey', SUZIE, position=[0, 2, 0.5], rotate=[90, 0, 0], scale=3, color=[255, 0, 0], bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[1, -2, 0], rotate=[0, 0, 45], color=[0, 255, 0], reflectivity=0, ior=1.3, alpha=0.3, bvh=USE_BVH)
plane = Object('Plane', PLANE, position=[0, 0, -0.3], scale=7, rotate=[100, 0, 0], color=[0, 0, 0], reflectivity=1, bvh=USE_BVH)
objects = [suzie, cube, plane]

icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 0, 0], ior=1.3, reflectivity=1, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[2, 0, 0], scale=0.5, color=[0, 0, 0], reflectivity=1, ior=1.3, bvh=USE_BVH)
# objects = [teapot, icosphere]

spherical_1 = Light('Spherical 1', position=[-3, -3, 1], intensity=1)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)

params = SceenParams(100, 100, 10, USE_BVH)
cam = Camera(position=[0, -5, 0], rotation=[90, 0, 180], fov=90)
skybox = Skybox('../littleengine/textures/puresky.jpg')

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
image, render_time = render_experimental(params, cam, skybox, objects, lights)
print(f'Total Render Time: {render_time:.3f}s')
image.show()
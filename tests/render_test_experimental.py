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

USE_BVH = False

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], color=[255, 0, 0], reflectivity=0.3, bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, bvh=USE_BVH)
icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, reflectivity=0.2, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[2, 0, 0], scale=0.5, color=[255, 0, 0], reflectivity=0.3, ior=1.3, bvh=USE_BVH)
objects = [suzie, cube]
# objects = [teapot, icosphere]

spherical_1 = Light('Spherical 1', position=[0, 3, 3], intensity=1.0)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
params = SceenParams(100, 100, [6, 20, 77], 3, USE_BVH)
cam = Camera(position=[0, 0, 5], rotation=[0, 180, 0], fov=90, aspect_ratio=1)

# print(Skybox('../littleengine/textures/miramar.jpeg').texture.shape)

image, render_time = render_experimental(cam, params, objects, lights)
print(f'Total Render Time: {render_time}')
image.show()
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

# suzie = Object('Monkey', SUZIE, position=[0, 2, 0.5], rotate=[90, 0, 0], scale=1, color=[255, 0, 0], bvh=USE_BVH)
# cube = Object('Cube', CUBE, position=[1, -2, 0.5], rotate=[0, 0, 45], color=[0, 255, 0], reflectivity=0, ior=1.3, alpha=0.3, bvh=USE_BVH)
# suzie = Object('Monkey', SUZIE, position=[1, 2, 0.55], rotate=[53.5, -10.5, 0], color=[255, 0, 0])

teapot = Object('Teapot', TEAPOT, position=[12, -5.5, 0], rotate=[90, 0, 90], scale=0.35, color=[134, 229, 231], reflectivity=0.25)
icosphere = Object('Icosphere', ICOSPHERE, position=[5, -2, 1], rotate=[90, 0, 90], color=[134, 229, 231], reflectivity=0.25)
cube1 = Object('Cube 1', CUBE, position=[-0.75, -2, 1], rotate=[0, 0, 45], scale=1.6, color=[0, 0, 0], reflectivity=1)
cube2 = Object('Cube 2', CUBE, position=[1.5, -4, 1], color=[231, 231, 231])
cube3 = Object('Cube 3', CUBE, position=[2, 1, 1], color=[231, 231, 231])
cube4 = Object('Cube 4', CUBE, position=[-2, -1, 1], color=[231, 231, 231])
cube5 = Object('Cube 5', CUBE, position=[-2, 1, 1], color=[231, 231, 231])
plane = Object('Plane', PLANE, position=[0, 0, 0], scale=7, rotate=[90, 0, 0], color=[231, 231, 231])
objects = [teapot, cube1, cube2, cube3, cube4, cube5, plane]

# objects = [teapot, icosphere]

spherical_1 = Light('Spherical 1', position=[0, 0, 5], intensity=1)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)

params = SceenParams(100, 100, 5)
# cam = Camera(position=[3, 0, 5], rotation=[145, 0, -90], fov=90)
# cam = Camera(position=[0, 0, 15], rotation=[180, 0, 0], fov=70)
cam = Camera(position=[0, 7, 1], rotation=[90, 0, 0], fov=50)
skybox = Skybox('../littleengine/textures/puresky.jpg')

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
image, render_time = render_experimental(params, cam, skybox, objects, lights)
print(f'Total Render Time: {render_time:.3f}s')
image.show()
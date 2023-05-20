import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np
from PIL import Image

import littleengine.mesh as mesh
import littleengine.object as object
import littleengine.camera as camera
import littleengine.render2 as render2
from littleengine.bvh import BVH, Bounding_Box, gen_meshlet_tree, build_meshlet_bounds
from littleengine.meshlet import meshlet_gen
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

# objects = []
suzie = object.Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
suzie.translate(0, 0, 4)
# objects.append(suzie)

cube = object.Object('Cube', CUBE)
cube.material_type = 'diffuse'
cube.translate(0, 0, 0)
cube.color = np.array([0, 255, 0])
# objects.append(cube)

ico2 = object.Object('Light', ICOSPHERE)
ico2.material_type = 'emissive'
ico2.translate(0, 0, 3)

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

o = ico2
meshlets = meshlet_gen(o)
objects = [mesh.Mesh(o.vertices, o.faces[meshlets[0].triangles], o.normals) for m in meshlets]

tree = gen_meshlet_tree(meshlets)
build_meshlet_bounds(o, meshlets, tree)

render2.render(w, h, cam, tree, objects, lights)
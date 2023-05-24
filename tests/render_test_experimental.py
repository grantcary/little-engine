import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from graphviz import Digraph

from littleengine import *
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'

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

ico = object.Object('Icosphere', ICOSPHERE)
ico.material_type = 'diffuse'
ico.translate(0, 0, 0)

pot = object.Object('Teapot', TEAPOT)
pot.material_type = 'diffuse'
pot.translate(0, 0, 3)

w, h = 100, 100
cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 5])
cam.rotation = np.array([0, 180, 0])
primary_rays = cam.primary_rays(w, h)

o = ico
st = time.time()
meshlets = meshlet_gen(o)
# meshlets = meshlets[len(meshlets) // 2:]
meshlets.sort(key=lambda meshlet: meshlet.centroid[2])
for i, m in enumerate(meshlets):
    m.index = i

print('Meshlet Gen:', time.time() - st)
objects = [mesh.Mesh(o.vertices, o.faces[m.triangles], o.normals) for m in meshlets]

tree = bounding_volume_hierarchy(o, meshlets)
# tree = gen_meshlet_tree(meshlets)
# build_meshlet_bounds(o, meshlets, tree)

def add_nodes_edges(tree, dot=None):
    if dot is None:
        dot = Digraph()
        dot.node(name=str(tree), label=str(tree.bounding_box.bounds))

    if tree.left:
        dot.node(name=str(tree.left) ,label=str(tree.left.bounding_box.bounds))
        dot.edge(str(tree), str(tree.left))
        dot = add_nodes_edges(tree.left, dot=dot)
    
    if tree.right:
        dot.node(name=str(tree.right) ,label=str(tree.right.bounding_box.bounds))
        dot.edge(str(tree), str(tree.right))
        dot = add_nodes_edges(tree.right, dot=dot)

    return dot

# dot = add_nodes_edges(tree)
# dot.render('binary_tree.gv', view=True)

render_experimental(w, h, cam, tree, objects)
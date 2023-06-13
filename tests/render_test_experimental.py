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

suzie = Object('Monkey', SUZIE, position=[0, 0, 0], color=[255, 0, 0], reflectivity=0.3)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)
objects = [suzie]

cam = Camera(position=[0, 0, 5], rotation=[0, 180, 0], fov=90, aspect_ratio=1)

o = suzie
st = time.time()
meshlets = simple_meshlet_gen(o)
print('Meshlet Gen:', time.time() - st)

print(len(np.unique([t for m in meshlets for t in m.triangles])) == o.faces.shape[0])

# meshlets = meshlets[len(meshlets) // 2:]
meshlets.sort(key=lambda meshlet: meshlet.centroid[2])
for i, m in enumerate(meshlets):
    m.index = i

objects = [Mesh(o.vertices, o.faces[m.triangles], o.normals) for m in meshlets]

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

# render_experimental(w, h, cam, tree, objects)
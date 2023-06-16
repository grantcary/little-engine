import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine import *
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'

def get_tree(object):
    meshlets = meshlet_gen(object)
    meshlets.sort(key=lambda meshlet: meshlet.centroid[2])
    for i, m in enumerate(meshlets):
        m.index = i
    return bounding_volume_hierarchy(object, meshlets)

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], color=[255, 0, 0], reflectivity=0.3)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)
# icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)
# teapot = Object('Teapot', TEAPOT, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)

suzie.bvh = get_tree(suzie)
cube.bvh = get_tree(cube)

objects = [suzie, cube]

spherical_1 = Light('Spherical 1', position=[0, 3, 3], intensity=1.0)
lights = [spherical_1]

cam = Camera(position=[0, 0, 5], rotation=[0, 180, 0], fov=90, aspect_ratio=1)

# dot = tools.add_nodes_edges(cube.bvh)
# dot.render('binary_tree.gv', view=True)

render_experimental(100, 100, cam, objects, lights)
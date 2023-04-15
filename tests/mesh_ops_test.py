import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.mesh as sm
import littleengine.mesh_ops as mo
import littleengine.read_file as rf
import tools

import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

obj = rf.OBJ()
vertices, faces = obj.read(OBJ_FILE)

o = sm.Mesh(vertices, faces)
# print(o.vertices)
o.set_normals()
o.triangulate()
print(o.faces)

# tools.plot_vectors_3D(o.normals)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.simple_mesh as sm
import littleengine.mesh_ops as mo
import littleengine.read_file as rf

import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

obj = rf.OBJ()
vertices, faces = obj.read(OBJ_FILE)

o = sm.Mesh(vertices, faces)

face = np.take(o.vertices, o.faces, 0)

AB = face[:,1] - face[:,0]
AC = face[:,2] - face[:,0]
normal = np.cross(AB, AC)
unit = normal / np.linalg.norm(normal, axis=1)[:,np.newaxis]
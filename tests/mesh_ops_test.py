import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.mesh as sm
import littleengine.read_file as rf
import tools

OBJ_FILE = '../test_objects/default_cube.obj'

obj = rf.OBJ()
vertices, faces = obj.read(OBJ_FILE)

m = sm.Mesh(vertices, faces)
# print(m.vertices)
m.set_normals()
m.triangulate()
print(m.faces)

# tools.plot_vectors_3D(m.normals)
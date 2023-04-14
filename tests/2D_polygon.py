import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.mesh as sm
import littleengine.mesh_ops as mo
import littleengine.read_file as rf
import tools
import ear

import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

def signed_area(polygon):
    area = 0
    for i in range(len(polygon)):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % len(polygon)]
        area += (v1[0] * v2[1] - v1[1] * v2[0])
    return 0.5 * area

obj = rf.OBJ()
vertices, faces = obj.read(OBJ_FILE)

# poly_v = np.array([[-1, 1, 0], [1, 1, 0], [1, 0, 1], [1, -1, 0], [-1, -1, 0], [-1, 0, 1]])
# poly_f = np.array([[0, 1, 2, 5], [5, 2, 3, 4]])
poly = sm.Mesh(vertices, faces)
poly.set_normals()

t = []
for i, f in enumerate(poly.faces):
    v = poly.vertices[f]
    dominant_axis = np.argmax(np.abs(poly.normals[i]))
    V_2D = np.delete(v, dominant_axis, axis=1)
    
    area = signed_area(V_2D)
    V_2D = V_2D[::-1] if area < 0 else V_2D
    
    # print(V_2D)

    triangles = ear.ear_clipping_triangulation_np(V_2D)
    for triangle in triangles:
        t.append(f[list(triangle)])
print(np.array(t))
# poly.faces = np.array(t)
# poly.set_normals()
# print(poly.normals)

# tools.plot_points_3D(poly.vertices, labels=True)
# tools.plot_vectors_3D(poly.normals)
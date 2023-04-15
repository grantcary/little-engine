import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.mesh as sm
import littleengine.mesh_ops as mo
import littleengine.read_file as rf
import tools
import ear

import numpy as np

OBJ_FILE = '../test_objects/suzie.obj'

def signed_area(polygon):
    area = 0
    for i in range(len(polygon)):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % len(polygon)]
        area += (v1[0] * v2[1] - v1[1] * v2[0])
    return 0.5 * area

def poly_to_tris(poly):
    t = []
    for i, f in enumerate(poly.faces):
        if poly.faces[i].shape[0] > 3:
            v = poly.vertices[f]
            dominant_axis = np.argmax(np.abs(poly.normals[i]))
            V_2D = np.delete(v, dominant_axis, axis=1)
            
            area = signed_area(V_2D)
            V_2D = V_2D[::-1] if area < 0 else V_2D

            triangles = ear.ear_clipping_triangulation_np(V_2D)
            for triangle in triangles:
                t.append(f[list(triangle)])
        else:
            t.append(f)
        
    return np.array(t)

# poly_v = np.array([[-1, 1, 0], [1, 1, 0], [1, 0, 1], [1, -1, 0], [-1, -1, 0], [-1, 0, 1]])
# poly_f = np.array([[0, 1, 2, 5], [5, 2, 3, 4]])

obj = rf.OBJ()
vertices, faces = obj.read(OBJ_FILE)

mesh = sm.Mesh(vertices, faces)
mesh.set_normals()

mesh.faces = poly_to_tris(mesh)
if len(mesh.faces[0]) > 3:
    print(True)
    mesh.set_normals()
print(mesh.faces)

# print(poly.normals)

# # tools.plot_points_3D(poly.vertices, labels=True)
# tools.plot_vectors_3D(poly.normals)
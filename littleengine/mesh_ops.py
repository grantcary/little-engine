import numpy as np

from littleengine.mesh import Vertex, Face, Mesh

def translate(m: Mesh, x, y, z):
    for v in m.vertices:
        v.coords += np.array([x, y, z])

def normal(f: Face):
    AB = f.vertices[1].coords - f.vertices[0].coords
    AC = f.vertices[2].coords - f.vertices[0].coords
    return np.cross(AB, AC)

def is_triangle(f: Face):
    return len(f.vertices) == 3
import numpy as np
from scipy.spatial import cKDTree

class Meshlet():
    def __init__(self, triangle_indices):
        self.triangle_indices = triangle_indices

def generate_triangle_adjacency(object):
    counts = np.full((object.vertices.shape[0]), 0, dtype=int)
    np.add.at(counts, object.faces, 1)

    offsets = np.full((object.vertices.shape[0]), 0, dtype=int)
    offset = 0
    for i in range(object.vertices.shape[0]):
        offsets[i] = offset
        offset += counts[i]

    data = np.full((object.faces.shape[0] * 3), 0, dtype=int)
    for i, face in enumerate(object.faces):
        data[offsets[face[0]]] = i
        data[offsets[face[1]]] = i
        data[offsets[face[2]]] = i

        offsets[face[0]] += 1
        offsets[face[1]] += 1
        offsets[face[2]] += 1

    for i in range(object.vertices.shape[0]):
        offsets[i] -= counts[i]

    return counts, offsets, data

def compute_triangle_cones(vertices, indices): 
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2, axis=1)
    
    area = np.sqrt(np.sum(normal ** 2, axis=1))
    invarea = np.where(area == 0.0, 0.0, 1.0 / area)

    triangles = v0 + v1 + v2 / 3.0
    normals = normal * invarea[:, np.newaxis]
    
    mesh_area = sum(area)
        
    return triangles.reshape(indices.shape), normals, mesh_area
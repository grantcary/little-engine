import time
from math import sqrt

import numpy as np
from scipy.spatial import cKDTree

class Meshlet():
    def __init__(self, triangle_indices = np.array([], dtype=int), centroid = None, normal = None):
        self.triangle_indices = triangle_indices
        self.centroid = centroid
        self.normal = normal

def generate_triangle_adjacency(object):
    counts = np.bincount(object.faces.flatten(), minlength=object.vertices.shape[0])
    offsets = np.hstack(([0], np.cumsum(counts[:-1])))

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

def get_neighbors(object, triangle_index, counts, offsets, data):
    vertices = object.faces[triangle_index].flatten()
    data_slices = [data[start:start+count] for start, count in zip(offsets[vertices], counts[vertices])]
    adjacent_triangles = np.concatenate(data_slices)
    adjacent_triangles = np.unique(adjacent_triangles)
    adjacent_triangles = np.delete(adjacent_triangles, np.argwhere(adjacent_triangles == triangle_index))
    return adjacent_triangles

def meshlet_gen(object, max_vertices=64, max_triangles=126):
    counts, offsets, data = generate_triangle_adjacency(object)
    available_triangles = np.arange(object.faces.shape[0])
    used_triangles = np.full(object.faces.shape[0], 0, dtype=int)

    while available_triangles.shape[0] > 0:
        print(available_triangles.shape[0])
        start_triangle = np.random.choice(available_triangles)

        neighbors = get_neighbors(object, start_triangle, counts, offsets, data)

        # delete neighboring triangles that aren't available any more
        st = neighbors.shape[0]
        delete = []
        for i in range(neighbors.shape[0]):
           if neighbors[i] not in available_triangles:
               delete.append(i)
    
        if len(delete) > 0:
            neighbors = np.delete(neighbors, np.array(delete))
            print(neighbors.shape[0]-st)

        # if neighbors.shape[0] > 0:
        #     # Sort neighboring triangles so that the ones sharing more vertices with the starting triangle are higer in the array 
        #     start_verts = object.faces[start_triangle]
        #     best = []
        #     count = []
        #     for i in range(neighbors.shape[0]):
        #         neighbor_verts = object.faces[neighbors[i]]
        #         common_verts = np.intersect1d(start_verts, neighbor_verts)
        #         count.append(common_verts.shape[0])

        #         if len(best) == 0:
        #             best.append(i)
        #         elif common_verts.shape[0] >= count[best[0]]:
        #             best.insert(0, i)
        #         else:
        #             best.append(i)

        #     best_neighbors = np.array(best)

        #     available_triangles = np.delete(available_triangles, neighbors)
        print(neighbors)
        available_triangles = np.delete(available_triangles, np.where(np.isin(available_triangles, neighbors)))
        available_triangles = np.delete(available_triangles, start_triangle)
import time
from math import sqrt

import numpy as np
from scipy.spatial import cKDTree

# Algorithm based off of https://github.com/zeux/meshoptimizer
# https://github.com/zeux/meshoptimizer/blob/master/src/clusterizer.cpp

class Meshlet():
    def __init__(self, triangles = np.array([], dtype=int)):
        self.triangles = triangles
        self.centroid = np.array([0, 0, 0], dtype=float)
        self.normal = np.array([0, 0, 0], dtype=float)

    def compute_cone(self):
        n = len(self.triangles)
        self.centroid *= 0.0 if n == 0 else 1.0 / float(n)
        axis_length = np.sum(self.normal**2)
        self.normal *= 0.0 if axis_length == 0.0 else 1.0 / sqrt(axis_length)

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

def get_neighbors(object, triangles, counts, offsets, data):
    vertices = object.faces[triangles].flatten()
    data_slices = [data[start:start+count] for start, count in zip(offsets[vertices], counts[vertices])]
    adjacent_triangles = np.unique(np.concatenate(data_slices))
    adjacent_triangles = np.delete(adjacent_triangles, np.argwhere(adjacent_triangles == triangles))
    return adjacent_triangles

def compute_triangle_cones(object): 
    v0 = object.vertices[object.faces[:, 0]]
    v1 = object.vertices[object.faces[:, 1]]
    v2 = object.vertices[object.faces[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0, axis=1)
    area = np.sqrt(np.sum(normal ** 2, axis=1))
    invarea = np.where(area == 0.0, 0.0, 1.0 / area)
    centroids = (v0 + v1 + v2 / 3.0).reshape(object.faces.shape)
    normals = normal * invarea[:, np.newaxis]
    area = np.sum(area)
    return centroids, normals, area

def topological_score(object, neighbors, live_triangles):
    return np.sum(live_triangles[object.faces[neighbors]], axis=1) - 3

def geographical_score(meshlet, neighbors, centroids, normals, cone_weight, expected_radius):
    distances2 = np.sum(np.square(centroids[neighbors] - meshlet.centroid), axis=1)
    spreads = np.sum(normals[neighbors] * meshlet.normal, axis=1)
    cone = 1.0 - spreads * cone_weight
    cone_clamped = np.where(cone < 1e-3, 1e-3, cone)
    return (1 + np.sqrt(distances2) / expected_radius * (1 - cone_weight)) * cone_clamped

def filter_scores(object, neighbors, scores, live_triangles, emitted_triangles, used_vertices):
    extras = np.sum((used_vertices[object.faces[neighbors]] == False) & (live_triangles[object.faces[neighbors]] != 1), axis=1)
    extras[extras != 0] += 1

    if scores.size == 0 or extras.size == 0: 
        return np.uint32(~0), 0

    min_score, min_extras = np.min(scores), np.min(extras)
    
    for idx, (score, extra) in enumerate(zip(scores, extras)):
        if score == min_score and extra == min_extras and emitted_triangles[neighbors[idx]] == False:
            return neighbors[idx], extra

    return np.uint32(~0), 0

def generate_kdtree(emitted_triangles, centroids):
    valid_triangle_indices = np.nonzero(emitted_triangles == False)[0]
    valid_centroids = centroids[valid_triangle_indices]
    tree = cKDTree(valid_centroids)
    return valid_triangle_indices, tree

def search_kdtree(tree, centroid, emitted_triangles, k):
    while True:
        indices = tree.query(centroid, k=k)[1]
        if isinstance(indices, np.ndarray):
            for index in indices:
                if emitted_triangles[index] == False:
                    return index
        else:
            if emitted_triangles[indices] == False:
                return indices
        k += 1
        if k > len(emitted_triangles):
            return None

def meshlet_gen(object, max_vertices=64, max_triangles=126, cone_weight=0.0):
    meshlets = []
    total_triangles = object.faces.shape[0]

    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, mesh_area = compute_triangle_cones(object)

    triangle_average_area = 0.0 if object.faces.shape[0] == 0 else mesh_area / float(object.faces.shape[0]) * 0.5
    meshtlet_expected_radius = sqrt(triangle_average_area * max_triangles) * 0.5

    live_triangles = counts.copy()
    used_triangles = np.full(len(object.faces), False, dtype=bool)
    used_vertices = np.full(len(object.vertices), False, dtype=bool)

    while total_triangles > 0:
        meshlet = Meshlet()
        valid_triangle_indices, tree = generate_kdtree(used_triangles, centroids)

        vertex_count = 0
        while total_triangles > 0 and vertex_count < max_vertices and len(meshlet.triangles) < max_triangles:
            meshlet.compute_cone()

            best_triangle = np.uint32(~0)

            if meshlet.triangles.shape[0] != 0:
                neighbors = get_neighbors(object, meshlet.triangles, counts, offsets, data)

                scores = geographical_score(meshlet, neighbors, centroids, normals, cone_weight, meshtlet_expected_radius)
                best_triangle, best_extra = filter_scores(object, neighbors, scores, live_triangles, used_triangles, used_vertices)
                
                if best_triangle != np.uint32(~0) and (vertex_count + best_extra > max_vertices or meshlet.triangles.shape[0] >= max_triangles):
                    scores = topological_score(object, neighbors, live_triangles)
                    best_triangle = filter_scores(object, neighbors, scores, live_triangles, used_triangles, used_vertices)[0]
                    if best_triangle != np.uint32(~0) and best_triangle is not None:
                        used_triangles[best_triangle] = True

            if best_triangle == np.uint32(~0):
                best_triangle = valid_triangle_indices[search_kdtree(tree, meshlet.centroid, used_triangles[valid_triangle_indices], 1)]
                if best_triangle is not None:
                    used_triangles[best_triangle] = True

            if best_triangle == np.uint32(~0):
                continue

            best_vertices = object.faces[best_triangle]

            used_extra = np.sum(used_vertices[best_vertices] == False)
            used_vertices[np.where(used_vertices[best_vertices] == False)] = True # used_vertices[best_vertices] = True (for a smaller meshlet count). current implementation creates pseudo uniform triangle per meshlet count
            used_triangles[best_triangle] = True

            meshlet.triangles = np.append(meshlet.triangles, best_triangle)

            neighbors = get_neighbors(object, meshlet.triangles, counts, offsets, data)
            if any(neighbors == best_triangle):
                counts[best_vertices] -= 1

            valid_triangle_indices, tree = generate_kdtree(used_triangles, centroids)

            meshlet.centroid += centroids[best_triangle]
            meshlet.normal += normals[best_triangle]
            
            live_triangles[best_vertices] -= 1
            total_triangles -= 1
            vertex_count += used_extra

        used_vertices[object.faces[meshlet.triangles].flatten()] = False
        meshlets.append(meshlet)
    return meshlets

def simple_meshlet_gen(object, max_vertices=64, max_triangles=126):
    meshlets = []
    total_triangles = object.faces.shape[0]

    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, _ = compute_triangle_cones(object)

    used_triangles = np.full(len(object.faces), False, dtype=bool)
    used_vertices = np.full(len(object.vertices), False, dtype=bool)

    while total_triangles > 0:
        meshlet = Meshlet()
        valid_triangle_indices, tree = generate_kdtree(used_triangles, centroids)

        vertex_count = 0
        while total_triangles > 0 and vertex_count < max_vertices and len(meshlet.triangles) < max_triangles:
            meshlet.compute_cone()

            best_triangle = valid_triangle_indices[search_kdtree(tree, meshlet.centroid, used_triangles[valid_triangle_indices], 1)]
            best_vertices = object.faces[best_triangle]

            used_extra = np.sum(used_vertices[best_vertices] == False)
            used_vertices[np.where(used_vertices[best_vertices] == False)] = True
            used_triangles[best_triangle] = True

            meshlet.triangles = np.append(meshlet.triangles, best_triangle)

            neighbors = get_neighbors(object, meshlet.triangles, counts, offsets, data)
            if any(neighbors == best_triangle):
                counts[best_vertices] -= 1

            valid_triangle_indices, tree = generate_kdtree(used_triangles, centroids)

            meshlet.centroid += centroids[best_triangle]
            meshlet.normal += normals[best_triangle]
            
            total_triangles -= 1
            vertex_count += used_extra

        used_vertices[object.faces[meshlet.triangles].flatten()] = False
        meshlets.append(meshlet)    
    return meshlets
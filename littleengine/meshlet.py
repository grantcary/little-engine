import time
from math import sqrt

import numpy as np
from scipy.spatial import cKDTree

class Meshlet():
    def __init__(self, triangle_indices = np.array([], dtype=int), centroid = None, normal = None):
        self.triangle_indices = triangle_indices
        self.vertex_count = 0
        self.centroid = centroid
        self.normal = normal

    def compute_cone(self):
        triangle_count = len(self.triangle_indices)

        center_scale = 0.0 if triangle_count == 0 else 1.0 / float(triangle_count)
        self.centroid *= center_scale

        axis_length = np.sum(self.normal * self.normal)
        axis_scale = 0.0 if axis_length == 0.0 else 1.0 / sqrt(axis_length)
        self.normal *= axis_scale

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

def compute_triangle_cones(object): 
    v0 = object.vertices[object.faces[:, 0]]
    v1 = object.vertices[object.faces[:, 1]]
    v2 = object.vertices[object.faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2, axis=1)
    
    area = np.sqrt(np.sum(normal ** 2, axis=1))
    invarea = np.where(area == 0.0, 0.0, 1.0 / area)

    centroids = (v0 + v1 + v2 / 3.0).reshape(object.faces.shape)
    normals = normal * invarea[:, np.newaxis]
    
    mesh_area = np.sum(area)
        
    return centroids, normals, mesh_area

def get_meshlet_score(distance2, spread, cone_weight, expected_radius):
    cone = 1.0 - spread * cone_weight
    # cone_clamped = 1e-3 if cone < 1e-3 else cone
    cone_clamped = np.where(cone < 1e-3, 1e-3, cone)

    return (1 + np.sqrt(distance2) / expected_radius * (1 - cone_weight)) * cone_clamped

def topological_score(object, neighbors, live_triangles):
    return np.sum(live_triangles[object.faces[neighbors]]) - 3

def geographical_score(meshlet, neighbors, centroids, normals, cone_weight, expected_radius):
    distances2 = np.sum(np.square(centroids[neighbors] - meshlet.centroid), axis=1)
    spreads = np.sum(normals[neighbors] * meshlet.normal, axis=1)
    cone = 1.0 - spreads * cone_weight
    cone_clamped = np.where(cone < 1e-3, 1e-3, cone)
    return (1 + np.sqrt(distances2) / expected_radius * (1 - cone_weight)) * cone_clamped

def filter_scores(object, neighbors, scores, live_triangles, used_vertices):
    best_triangles = np.full(neighbors.shape[0], np.uint32(~0), dtype=np.uint32)

    extras = np.sum(used_vertices[object.faces[neighbors]], axis=1)
    has_live_triangles = np.any(live_triangles[object.faces[neighbors]] == 1, axis=1)
    extras[has_live_triangles] = 0
    extras += 1

    best_index = np.argmin(np.stack((extras, scores)))
    return None if extras[best_index] == np.inf else neighbors[best_index]

def compute_triangle_scores(object, meshlet, neighbor, live_triangles, used_vertices, best_triangle, best_extra, best_score, expected_radius, cone_weight, centroids, normals, topo_priority=False):
    extra = np.sum(used_vertices[object.faces[neighbor]])

    if extra != 0:
        if any(live_triangles[object.faces[neighbor]] == 1):
            extra = 0
        extra += 1

    if extra > best_extra:
        return best_triangle, best_extra, best_score

    if topo_priority:
        score = np.sum(live_triangles[object.faces[neighbor]]) - 3
    else:
        distances2 = np.sum(np.square(centroids[neighbor] - meshlet.centroid))
        spreads = np.sum(normals[neighbor] * meshlet.normal)
        score = get_meshlet_score(distances2, spreads, cone_weight, expected_radius)

    if extra < best_extra or score < best_score:
        return neighbor, extra, score
        
    return best_triangle, best_extra, best_score

def meshlet_gen(object, max_vertices=64, max_triangles=126):
    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, mesh_area = compute_triangle_cones(object)

    cone_weight = 0.0
    triangle_average_area = 0.0 if object.faces.shape[0] == 0 else mesh_area / float(object.faces.shape[0]) * 0.5
    meshtlet_expected_radius = sqrt(triangle_average_area * max_triangles) * 0.5

    cones = np.hstack((centroids, normals))
    tree = cKDTree(cones)

    meshlets = []
    available_triangles = np.arange(object.faces.shape[0])
    live_triangles = counts.copy()
    used_vertices = np.full(len(object.vertices), -1, dtype=int)



    while len(available_triangles) > 0:
        start_triangle = np.random.choice(available_triangles)

        # Init mesh object
        meshlet = Meshlet()
        meshlet.triangle_indices = np.append(meshlet.triangle_indices, start_triangle)
        meshlet.vertex_count += 3
        meshlet.centroid = centroids[start_triangle]
        meshlet.normal = normals[start_triangle]
        meshlet.compute_cone()

        num_vertices = 0
        neighbor_index = 0
        neighbors = get_neighbors(object, start_triangle, counts, offsets, data)

        best_triangle = np.uint32(~0)
        best_extra = 5
        best_score = np.float32(np.finfo(np.float32).max)

        # Add triangles to meshlet until it is full
        while len(available_triangles) > 0 and num_vertices < max_vertices and len(meshlet.triangle_indices) < max_triangles:
            scores = geographical_score(meshlet, neighbors, centroids, normals, cone_weight, meshtlet_expected_radius)
            best_triangle = filter_scores(object, neighbors, scores, live_triangles, used_vertices)
            
            if best_triangle != np.uint32(~0):
                scores = topological_score()


            if best_triangle != np.uint32(~0) and (object.vertices.shape[0] + best_extra > max_vertices or object.faces.shape[0] >= max_triangles):
                best_triangle, best_extra, best_score = compute_triangle_scores(object, meshlet, neighbors[neighbor_index], live_triangles, used_vertices, best_triangle, best_extra, best_score, meshtlet_expected_radius, cone_weight, centroids, normals, True)
                print('Geo Priority: ', best_triangle, best_extra, best_score)

            if best_triangle == np.uint32(~0):

                index = np.uint32(~0)
                limit = np.float32(np.finfo(np.float32).max)

                meshlet_cone = np.hstack((meshlet.centroid, meshlet.normal))
                distance, index = tree.query(meshlet_cone)
                print(distance, index)
                print(object.faces[index])

                best_triangle = index

            if best_triangle == np.uint32(~0):
                continue

            used_extra = np.sum(used_vertices[object.faces[best_triangle]] == 1)
            result = False
            if object.vertices.shape[0] + used_extra > max_vertices or object.faces.shape[0] >= max_triangles:
                used_vertices[object.faces[meshlet.triangle_indices].flatten()] = 0
                result = True

            if result:
                meshlet.triangle_indices = np.append(meshlet.triangle_indices, best_triangle)

            live_triangles[object.faces[best_triangle]] -= 1

            if neighbors[neighbor_index] == best_triangle:
                counts[object.faces[best_triangle]] -= 1

            meshlet.centroid += centroids[best_triangle]
            meshlet.normal += normals[best_triangle]
            # meshlet.compute_cone()

            neighbor_index += 1

        meshlets.append(meshlet)


# def meshlet_gen(object, max_vertices=64, max_triangles=126):
#     counts, offsets, data = generate_triangle_adjacency(object)
#     centroids, normals, mesh_area = compute_triangle_cones(object)

#     cone_weight = 0.0
#     triangle_average_area = 0.0 if object.faces.shape[0] == 0 else mesh_area / float(object.faces.shape[0]) * 0.5
#     meshtlet_expected_radius = sqrt(triangle_average_area * max_triangles) * 0.5

#     cones = np.hstack((centroids, normals))
#     tree = cKDTree(cones)

#     meshlets = []
#     available_triangles = np.arange(object.faces.shape[0])
#     live_triangles = counts.copy()
#     used_vertices = np.full(len(object.vertices), -1, dtype=int)

#     while len(available_triangles) > 0:
#         start_triangle = np.random.choice(available_triangles)

#         # Init mesh object
#         meshlet = Meshlet()
#         meshlet.triangle_indices = np.append(meshlet.triangle_indices, start_triangle)
#         meshlet.vertex_count += 3
#         meshlet.centroid = centroids[start_triangle]
#         meshlet.normal = normals[start_triangle]
#         meshlet.compute_cone()

#         num_vertices = 0
#         neighbor_index = 0
#         neighbors = get_neighbors(object, start_triangle, counts, offsets, data)

#         best_triangle = np.uint32(~0)
#         best_extra = 5
#         best_score = np.float32(np.finfo(np.float32).max)

#         # Add triangles to meshlet until it is full
#         while len(available_triangles) > 0 and num_vertices < max_vertices and len(meshlet.triangle_indices) < max_triangles:
#             best_triangle, best_extra, best_score = compute_triangle_scores(object, meshlet, neighbors[neighbor_index], live_triangles, used_vertices, best_triangle, best_extra, best_score, meshtlet_expected_radius, cone_weight, centroids, normals, False)
#             print('Topo Priority:', best_triangle, best_extra, best_score)

#             if best_triangle != np.uint32(~0) and (object.vertices.shape[0] + best_extra > max_vertices or object.faces.shape[0] >= max_triangles):
#                 best_triangle, best_extra, best_score = compute_triangle_scores(object, meshlet, neighbors[neighbor_index], live_triangles, used_vertices, best_triangle, best_extra, best_score, meshtlet_expected_radius, cone_weight, centroids, normals, True)
#                 print('Geo Priority: ', best_triangle, best_extra, best_score)

#             if best_triangle == np.uint32(~0):

#                 index = np.uint32(~0)
#                 limit = np.float32(np.finfo(np.float32).max)

#                 meshlet_cone = np.hstack((meshlet.centroid, meshlet.normal))
#                 distance, index = tree.query(meshlet_cone)
#                 print(distance, index)
#                 print(object.faces[index])

#                 best_triangle = index

#             if best_triangle == np.uint32(~0):
#                 continue

#             used_extra = np.sum(used_vertices[object.faces[best_triangle]] == 1)
#             result = False
#             if object.vertices.shape[0] + used_extra > max_vertices or object.faces.shape[0] >= max_triangles:
#                 used_vertices[object.faces[meshlet.triangle_indices].flatten()] = 0
#                 result = True

#             if result:
#                 meshlet.triangle_indices = np.append(meshlet.triangle_indices, best_triangle)

#             live_triangles[object.faces[best_triangle]] -= 1

#             if neighbors[neighbor_index] == best_triangle:
#                 counts[object.faces[best_triangle]] -= 1

#             meshlet.centroid += centroids[best_triangle]
#             meshlet.normal += normals[best_triangle]
#             # meshlet.compute_cone()

#             neighbor_index += 1

#         meshlets.append(meshlet)

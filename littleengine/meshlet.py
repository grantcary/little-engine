import time
import math

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
    
    mesh_area = sum(area)
        
    return centroids, normals, mesh_area

def compute_meshlet_cone(meshlet):
    triangle_count = meshlet.triangle_indices.shape[0]

    center_scale = 0.0 if triangle_count == 0 else 1.0 / float(triangle_count)
    meshlet.centroid *= center_scale

    axis_length = np.sum(meshlet.normal * meshlet.normal)
    axis_scale = 0.0 if axis_length == 0.0 else 1.0 / math.sqrt(axis_length)
    meshlet.normal *= axis_scale


def meshlet_gen(object, max_vertices=64, max_triangles=126):
    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, mesh_area = compute_triangle_cones(object)

    cones = np.hstack((centroids, normals))
    tree = cKDTree(cones)

    meshlets = []
    available_triangles = np.arange(object.faces.shape[0])
    
    meshlet_triangles = 0

    while len(available_triangles) > 0:
        start_triangle = np.random.choice(available_triangles)

        # Init mesh object
        meshlet = Meshlet()
        meshlet.triangle_indices = np.append(meshlet.triangle_indices, start_triangle)
        meshlet.centroid = centroids[start_triangle]
        meshlet.normal = normals[start_triangle]

        compute_meshlet_cone(meshlet)
        # print(centroids[start_triangle], normals[start_triangle])
        # print(meshlet.centroid, meshlet.normal)

        num_vertices = 0
        # Add triangles to meshlet until it is full
        while len(available_triangles) > 0 and num_vertices < max_vertices and len(meshlet.triangle_indices) < max_triangles:
            neighbors = get_neighbors(object, meshlet.triangle_indices, counts, offsets, data)

        #     # Compute score for each neighboring triangle
        #     scores = compute_triangle_scores(object, neighbors, meshlet_vertices, cone_origin, cone_axis, cone_angle, tree)

        #     # Select triangle with the highest score and add it to the meshlet
        #     selected_triangle = neighbors[np.argmax(scores)]
        #     meshlet.add_triangle(selected_triangle, object)
        #     meshlet_triangles.append(selected_triangle)

        #     # Update meshlet vertices
        #     for v in object.faces[selected_triangle]:
        #         meshlet_vertices.add(v)

        #     compute_meshlet_cone(meshlet)

        # # Add meshlet to list of meshlets
        # meshlets.append(meshlet)
        # available_triangles = np.delete(available_triangles, np.where(available_triangles == start_triangle))
        # # yield meshlet
        # meshlet_triangles.clear()
        # meshlet_vertices.clear()
        
        # # Initialize new meshlet with the next triangle
        # if len(counts) > 0:
        #     start_triangle = np.argmax(counts)
        #     cone_origin, cone_axis, cone_angle = compute_meshlet_cone(centroids[start_triangle], normals[start_triangle])
        #     meshlet_triangles.append(start_triangle)
        #     for v in object.faces[start_triangle]:
        #         meshlet_vertices.add(v)

        #     counts[start_triangle] = 0
        #     for neighbor in get_neighbors([start_triangle], data, counts, offsets):
        #         counts[neighbor] -= 1
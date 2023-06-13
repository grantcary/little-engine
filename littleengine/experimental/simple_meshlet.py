from littleengine.experimental.meshlet import Meshlet, generate_triangle_adjacency, get_neighbors, compute_triangle_cones, search_kdtree
from scipy.spatial import cKDTree
import numpy as np
import time

def simple_meshlet_gen(object, max_vertices=64, max_triangles=126, cone_weight=0.0):
    meshlets = []
    total_triangles = object.faces.shape[0]

    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, mesh_area = compute_triangle_cones(object)

    live_triangles = counts.copy()
    emitted_triangles = np.full(len(object.faces), 0, dtype=int)

    used_vertices = np.full(len(object.vertices), 0, dtype=int) # 0: unused, 1: used
    available_vertices = np.full(len(object.vertices), True, dtype=bool)

    meshlet_index = 0
    while total_triangles > 0:
        meshlet = Meshlet(meshlet_index)

        # st = time.time()
        valid_triangle_indices = np.nonzero(emitted_triangles == 0)[0]
        valid_centroids = centroids[valid_triangle_indices]
        tree = cKDTree(valid_centroids)
        # print('Tree Gen Time:', time.time() - st)

        vertex_count = 0
        while total_triangles > 0 and vertex_count < max_vertices and len(meshlet.triangles) < max_triangles:
            meshlet.compute_cone()

            print("Triangles Left:", sum(emitted_triangles == 0))
            if len(valid_triangle_indices) == 1:
                best_triangle = valid_triangle_indices[0]
            else:
                best_triangle_index = search_kdtree(tree, meshlet.centroid, emitted_triangles, 1)
                best_triangle = valid_triangle_indices[best_triangle_index]
            # print(best_triangle, best_triangle_index)
            best_vertices = object.faces[best_triangle]

            used_extra = np.sum(used_vertices[best_vertices] == 0)
            used_vertices[np.where(used_vertices[best_vertices] == 0)] = 1
            available_vertices[np.where(available_vertices[best_vertices] == True)] = False

            meshlet.triangles = np.append(meshlet.triangles, best_triangle)
            vertex_count += used_extra

            live_triangles[best_vertices] -= 1

            neighbors = get_neighbors(object, meshlet.triangles, counts, offsets, data)
            # print("Neighbors:", neighbors)
            if any(neighbors == best_triangle):
                counts[best_vertices] -= 1

            meshlet.centroid += centroids[best_triangle]
            meshlet.normal += normals[best_triangle]

            total_triangles -= 1
            emitted_triangles[best_triangle] = 1

            valid_triangle_indices = np.nonzero(emitted_triangles == 0)[0]
            valid_centroids = centroids[valid_triangle_indices]
            tree = cKDTree(valid_centroids)
        
        print("Meshlet Index:", meshlet.index, "Triangle Count:", meshlet.triangles.shape[0])
        meshlets.append(meshlet)
        meshlet_index += 1
        used_vertices[object.faces[meshlet.triangles].flatten()] = 0
    
    return meshlets

# def simple_meshlet_gen(object, max_vertices=64, max_triangles=126, cone_weight=0.0):
#     meshlets = []
#     total_triangles = object.faces.shape[0]

#     counts, offsets, data = generate_triangle_adjacency(object)
#     centroids, normals, mesh_area = compute_triangle_cones(object)

#     live_triangles = counts.copy()
#     emitted_triangles = np.full(len(object.faces), 0, dtype=int)

#     used_vertices = np.full(len(object.vertices), 0, dtype=int) # 0: unused, 1: used
#     available_vertices = np.full(len(object.vertices), True, dtype=bool)

#     meshlet_index = 0
#     while total_triangles > 0:
#         meshlet = Meshlet(meshlet_index)
        
#         st = time.time()
#         tree = cKDTree(centroids)
#         print('Tree Gen Time:', time.time() - st)

#         vertex_count = 0
#         while total_triangles > 0 and vertex_count < max_vertices and len(meshlet.triangles) < max_triangles:
#             meshlet.compute_cone()

#             best_triangle = search_kdtree(tree, meshlet.centroid, emitted_triangles, 1)
#             best_vertices = object.faces[best_triangle]

#             used_extra = np.sum(used_vertices[best_vertices] == 0)
#             print((used_vertices[best_vertices] == 0).shape, best_triangle)
#             used_vertices[np.where(used_vertices[best_vertices] == 0)] = 1
#             available_vertices[np.where(available_vertices[best_vertices] == True)] = False

#             meshlet.triangles = np.append(meshlet.triangles, best_triangle)
#             vertex_count += used_extra

#             live_triangles[best_vertices] -= 1

#             neighbors = get_neighbors(object, meshlet.triangles, counts, offsets, data)
#             if any(neighbors == best_triangle):
#                 counts[best_vertices] -= 1

#             meshlet.centroid += centroids[best_triangle]
#             meshlet.normal += normals[best_triangle]

#             total_triangles -= 1
#             emitted_triangles[best_triangle] = 1
        
#         print(meshlet.index)
#         meshlets.append(meshlet)
#         meshlet_index += 1
#         used_vertices[object.faces[meshlet.triangles].flatten()] = 0
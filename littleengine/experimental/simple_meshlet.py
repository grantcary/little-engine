from littleengine.experimental.meshlet import Meshlet, generate_triangle_adjacency, get_neighbors, compute_triangle_cones
from scipy.spatial import cKDTree
import numpy as np

def generate_kdtree(emitted_triangles, centroids):
    valid_triangle_indices = np.nonzero(emitted_triangles == 0)[0]
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

def simple_meshlet_gen(object, max_vertices=64, max_triangles=126):
    meshlets = []
    total_triangles = object.faces.shape[0]

    counts, offsets, data = generate_triangle_adjacency(object)
    centroids, normals, _ = compute_triangle_cones(object)

    used_triangles = np.full(len(object.faces), False, dtype=bool)
    used_vertices = np.full(len(object.vertices), False, dtype=bool)

    meshlet_index = 0
    while total_triangles > 0:
        meshlet = Meshlet(meshlet_index)
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
        
        print("Meshlet", meshlet.index, "Triangle Count:", meshlet.triangles.shape[0])

        meshlets.append(meshlet)
        used_vertices[object.faces[meshlet.triangles].flatten()] = False
        meshlet_index += 1
    
    return meshlets
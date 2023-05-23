import sys
import math
import time

import numpy as np
from PIL import Image
from numba import njit

# np.set_printoptions(threshold=sys.maxsize)

@njit
def ray_triangle_intersection(ray_origin, ray_directions, triangle_vertices):
    epsilon = 1e-6
    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_directions, edge2)
    a = np.sum(edge1 * h)

    parallel_mask = np.abs(a) < epsilon
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.sum(s * h)

    valid_u = (u >= 0) & (u <= 1)
    q = np.cross(s, edge1)
    v = f * np.sum(ray_directions * q)

    valid_v = (v >= 0) & (u + v <= 1)
    t = f * np.sum(edge2 * q)
    valid_t = t > epsilon

    intersection_mask = ~parallel_mask & valid_u & valid_v & valid_t
    intersection_points = ray_origin + (ray_directions * t).T
    
    return intersection_mask, intersection_points

def trace(objects, meshlet_indices, ray_origin, ray_directions):
    for enum_index, meshlet in enumerate([objects[i] for i in meshlet_indices]):
        triangles_vertices = meshlet.vertices[meshlet.faces]
        hit, intersection_points = ray_triangle_intersection(ray_origin, ray_directions, triangles_vertices)
        print(intersection_points)

def render(w, h, cam, bvh, objects):
    st = time.time()
    primary_rays = cam.primary_rays(w, h)
    print('Generate Primary Rays:', time.time() - st)
    
    st = time.time()
    trace(objects, np.array([0, 3]), cam.position, primary_rays[5261])
    print('Primary Ray Cast:', time.time() - st)   

    # st = time.time()
    # primary_mask = np.full(primary_rays.shape[0], -1, dtype=int)
    # for i, ray in enumerate(primary_rays):
    #     search = bvh.search_collision(cam.position, ray)
    #     row, col = i // w, i % w
    #     if search:
    #         primary_mask[i] = i
    #         # print(search, i)

    # mask = np.where(primary_mask != -1)
    # filtered_rays = primary_rays[mask]
    # print('Cull Rays:', time.time() - st)
    # print(len(primary_rays), len(primary_rays[mask]))


    # min_t_values, object_indices, _ = trace(objects, cam.position, filtered_rays)
    # object_i = np.full(primary_rays.shape[0], -1, dtype=int)
    # object_i[mask] = object_indices

    # img = np.full((h, w), 127, dtype=np.uint8)
    # for i, ray in enumerate(primary_rays):
    #     row, col = i // w, i % w
    #     img[row, col] = 255 if object_i[i] != -1 else 127

    # rendered_image = Image.fromarray(img, 'L')
    # rendered_image.show()

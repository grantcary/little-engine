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
    v0, v1, v2 = triangle_vertices

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_directions, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = np.abs(a) < epsilon
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h.T)

    valid_u = (u >= 0) & (u <= 1)
    q = np.cross(s, edge1)
    v = f * np.dot(ray_directions, q.T)

    valid_v = (v >= 0) & (u + v <= 1)
    t = f * np.dot(edge2, q.T)
    valid_t = t > epsilon

    intersection_mask = ~parallel_mask & valid_u & valid_v & valid_t
    intersection_points = ray_origin + ray_directions * t.reshape(-1, 1)
    
    return intersection_mask, intersection_points

def trace(objects, ray_origin, ray_directions):
    total_rays = ray_directions.shape[0]
    min_t_values = np.full(total_rays, np.inf)
    object_indices = np.full(total_rays, -1, dtype=int)
    tri_indices = np.full(total_rays, -1, dtype=int)

    for obj_index, obj in enumerate(objects):
        triangle_vertices = obj.vertices[obj.faces]

        for tri_index, triangle in enumerate(triangle_vertices):
            hit, intersection_points = ray_triangle_intersection(ray_origin, ray_directions, triangle)
            t_values = np.linalg.norm(intersection_points - ray_origin, axis=-1)
            hit = np.diagonal(hit) if len(hit.shape) > 1 else hit
            update_mask = (t_values < min_t_values) & hit
            min_t_values[update_mask] = t_values[update_mask]
            object_indices[update_mask] = obj_index
            tri_indices[update_mask] = tri_index

    return min_t_values, object_indices, tri_indices

def shade(objects, lights, intersection_points, object_indices):
    n = intersection_points.shape[0]
    hit_colors = np.zeros((n, 3))

    for light in lights:
        light_directions = light.position - intersection_points
        len2 = np.sum(light_directions * light_directions, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        shadow_ray_t, shadow_ray_indices, _ = trace(objects, intersection_points, normalized_light_directions)

        shadow_ray_len2 = shadow_ray_t * shadow_ray_t
        isInShadow = (shadow_ray_indices != -1) & (shadow_ray_len2 < len2)

        for i in range(n):
            obj = objects[object_indices[i]]
            if not isInShadow[i]:
                hit_colors[i] = obj.color * (1 - isInShadow[i])

    return hit_colors

def render(w, h, cam, bvh, objects, lights):
    st = time.time()
    primary_rays = cam.primary_rays(w, h)
    print('Generate Primary Rays:', time.time() - st)
    
    st = time.time()
    primary_mask = np.full(primary_rays.shape[0], -1, dtype=int)
    for i, ray in enumerate(primary_rays):
        search = bvh.search_collision(cam.position, ray)
        if search is not None:
            primary_mask[i] = i

    mask = np.where(primary_mask != -1)
    filtered_rays = primary_rays[mask]
    print('Cull Rays:', time.time() - st)

    st = time.time()
    min_t_values, object_indices, _ = trace(objects, cam.position, filtered_rays)
    print('Primary Ray Cast:', time.time() - st)   

    # img = np.full((w, h), 0, dtype=np.uint8)
    # st = time.time()
    # for i, ray in enumerate(primary_rays):
    #     row, col = i // w, i % w
    #     search = scene.search_collision(cam.position, ray)
    #     # print(search)
    #     img[row, col] = 255 if search != None else 127
    # print(time.time() - st)

    # rendered_image = Image.fromarray(img, 'L')
    # rendered_image.show()

    # for ray in primary_rays:
        

    # img = np.full((h, w, 3), [6, 20, 77] , dtype=np.uint8)
    # for i in range(primary_rays.shape[0]):
    #     row, col = i // w, i % w
    #     img[row, col] = objects[object_indices[i]].color if object_indices[i] != -1 else [6, 20, 77]

    # rendered_image = Image.fromarray(img, 'RGB')

    # rendered_image.show()
import sys
import math
import time
import random

import numpy as np
from PIL import Image
from numba import cuda, njit

np.set_printoptions(threshold=sys.maxsize)

def rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))

def camera_rays(w, h, cam):
    aspect_ratio = w / h
    angle = math.tan(math.radians(cam.fov / 2))
    camera_distance = 1 / angle

    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    normalized_x = np.interp(x_indices, (0, w-1), (-1, 1))
    normalized_y = np.interp(y_indices, (0, h-1), (1, -1))
    
    xx = normalized_x * angle * aspect_ratio
    yy = normalized_y * angle
    cc = np.full_like(xx, camera_distance)

    a = np.stack([xx, yy, cc], axis=-1)
    ray_vectors = a / np.linalg.norm(a, axis=-1, keepdims=True)

    rotation = rotation_matrix(cam.rotation)
    rotated_ray_vectors = np.dot(ray_vectors.reshape(-1, 3), rotation.T)

    return rotated_ray_vectors.reshape(-1, 3)

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

def render(w, h, cam, objects, lights):
    ray_vectors = camera_rays(w, h, cam)
    st = time.time()
    min_t_values, object_indices, _ = trace(objects, cam.position, ray_vectors)
    print('Primary Ray Cast:', time.time() - st)            

    intersection_points = cam.position + ray_vectors * min_t_values.reshape(-1, 1)
    valid_intersection_mask = object_indices != -1

    st = time.time()
    hit_colors = shade(objects, lights, intersection_points[valid_intersection_mask], object_indices[valid_intersection_mask])
    print('Diffuse Ray Cast:', time.time() - st)

    hit_color_image = np.full((h, w, 3), [6, 20, 77] , dtype=np.uint8)
    hit_color_image[valid_intersection_mask.reshape(h, w)] = hit_colors
    rendered_image = Image.fromarray(hit_color_image, 'RGB')

    rendered_image.show()
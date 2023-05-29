import sys
import time

import numpy as np
from PIL import Image
import progressbar

np.set_printoptions(threshold=sys.maxsize)

def ray_triangle_intersection(ray_origin, ray_directions, triangle_vertices):
    epsilon = 1e-6
    v0, v1, v2 = triangle_vertices

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_directions, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = np.abs(a) < epsilon

    f = np.zeros_like(a)
    non_zero_a_indices = np.abs(a) >= epsilon
    f[non_zero_a_indices] = 1.0 / a[non_zero_a_indices]
    
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

def trace(objects, ray_origin, ray_directions, background_color):
    total_rays = ray_directions.shape[0]
    min_t_values = np.full(total_rays, np.inf)
    object_indices = np.full(total_rays, -1, dtype=int)
    tri_indices = np.full(total_rays, -1, dtype=int)
    tri_normals = np.full((total_rays, 3), [0.0, 0.0, 0.0], dtype=float)
    color_values = np.full((total_rays, 3), background_color, dtype=float)
    reflectivity_values = np.full(total_rays, 0.0, dtype=float)

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
            tri_normals[update_mask] = obj.normals[tri_index] # check if this is the correct usage of a mask in this situation
            color_values[update_mask] = obj.color
            reflectivity_values[update_mask] = obj.reflectivity

    return min_t_values, object_indices, tri_indices, tri_normals, color_values, reflectivity_values

def shade(objects, lights, intersection_points, object_indices, background_color):
    n = intersection_points.shape[0]
    hit_colors = np.zeros((n, 3))

    for light in lights:
        light_directions = light.position - intersection_points
        len2 = np.sum(light_directions * light_directions, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        shadow_ray_t, shadow_ray_indices, _, _, _, _= trace(objects, intersection_points, normalized_light_directions, background_color)

        shadow_ray_len2 = shadow_ray_t * shadow_ray_t
        isInShadow = (shadow_ray_indices != -1) & (shadow_ray_len2 < len2)

        for i in range(n):
            obj = objects[object_indices[i]]
            if not isInShadow[i]:
                hit_colors[i] = obj.color * (1 - isInShadow[i])

    return hit_colors

def reflect(ray_origins, ray_directions, hit_normals, reflectivity_values, background_color, bias):
    ray_origins = ray_origins + hit_normals * bias
    ray_directions = ray_directions - 2 * np.einsum('ij,ij->i', ray_directions, hit_normals)[:, np.newaxis] * hit_normals
    hit_colors = reflectivity_values[:, np.newaxis] * 0.8 * background_color
    return ray_origins, ray_directions, hit_colors

# def refract():
#     pass

def calculate_scene(w, h, cam, objects, lights):
    background_color = [6, 20, 77]
    image_hit_colors = np.zeros((h * w, 3), dtype=np.float32)
    intersection_mask = None
    max_depth = 3

    primary_rays = cam.primary_rays(w, h)
    ray_origins, ray_directions = cam.position, primary_rays

    bar = progressbar.ProgressBar()
    for current_depth in bar(range(max_depth)):
        min_t_values, object_indices, triangle_indices, hit_normals, color_values, reflectivity_values = trace(objects, ray_origins, ray_directions, background_color)

        intersection_points = ray_origins + ray_directions * min_t_values.reshape(-1, 1)
        current_intersection_mask = object_indices != -1

        if intersection_mask is not None:
            previous_mask = np.full(h * w, False, dtype=bool)
            previous_mask[intersection_mask] = current_intersection_mask
            intersection_mask = previous_mask
        else:
            intersection_mask = current_intersection_mask
        
        ray_origins = intersection_points[current_intersection_mask]
        ray_directions = ray_directions[current_intersection_mask]
        hit_normals = hit_normals[current_intersection_mask]
        reflectivity_values = reflectivity_values[current_intersection_mask]
        color_values = color_values[current_intersection_mask]

        hit_colors = shade(objects, lights, ray_origins, object_indices[current_intersection_mask], background_color)
        image_hit_colors[intersection_mask] += hit_colors
        
        ray_origins, ray_directions, hit_colors = reflect(ray_origins, ray_directions, hit_normals, reflectivity_values, background_color, 1e-4)
        image_hit_colors[intersection_mask] += hit_colors

        bar.update(current_depth)

    no_hit_mask = np.all(image_hit_colors == 0, axis=-1)    
    image_hit_colors[no_hit_mask] += background_color

    image_hit_colors = np.clip(image_hit_colors, 0, 255).astype(np.uint8)
    return image_hit_colors

def render(w, h, cam, objects, lights):
    tst = time.time()
    # hit_color_image = 
    hit_color_image = calculate_scene(w, h, cam, objects, lights)
    print(f'\nTotal Render Time: {time.time() - tst:.4f}s')

    rendered_image = Image.fromarray(hit_color_image.reshape(h, w, 3), 'RGB')
    rendered_image.show()
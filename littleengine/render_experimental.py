import time

import numpy as np
from PIL import Image

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
    tri_normals = np.full((total_rays, 3), [0.0, 0.0, 0.0], dtype=float)
    color_values = np.full((total_rays, 3), background_color, dtype=float)
    reflectivity_values = np.full(total_rays, 0.0, dtype=float)
    ior_values = np.full(total_rays, 0.0, dtype=float)

    for obj_index, obj in enumerate(objects):
        triangle_vertices = obj.vertices[obj.faces]

        for tri_index, triangle in enumerate(triangle_vertices):
            hit, intersection_points = ray_triangle_intersection(ray_origin, ray_directions, triangle)
            t_values = np.linalg.norm(intersection_points - ray_origin, axis=-1)
            hit = np.diagonal(hit) if len(hit.shape) > 1 else hit
            update_mask = (t_values < min_t_values) & hit
            min_t_values[update_mask] = t_values[update_mask]
            object_indices[update_mask] = obj_index
            tri_normals[update_mask] = obj.normals[tri_index] # check if this is the correct usage of a mask in this situation
            color_values[update_mask] = obj.color
            reflectivity_values[update_mask] = obj.reflectivity
            ior_values[update_mask] = obj.ior

    return min_t_values, object_indices, tri_normals, color_values, reflectivity_values, ior_values

def shade(objects, lights, intersection_points, hit_normals, object_indices, background_color):
    n = intersection_points.shape[0]
    hit_colors = np.zeros((n, 3), dtype=np.float32)

    for light in lights:
        light_directions = light.position - intersection_points
        len2 = np.sum(light_directions * light_directions, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        shadow_ray_t, shadow_ray_indices, _, _, _, _ = trace(objects, intersection_points, normalized_light_directions, background_color)

        shadow_ray_len2 = shadow_ray_t * shadow_ray_t
        isInShadow = (shadow_ray_indices != -1) & (shadow_ray_len2 < len2)

        cos_theta = np.einsum('ij,ij->i', hit_normals, normalized_light_directions)
        for i in range(n):
            obj = objects[object_indices[i]]
            hit_colors[i] = obj.color * light.intensity * max(0, cos_theta[i]) * (1 - isInShadow[i])

    return hit_colors

def reflect(ray_origins, ray_directions, hit_normals, reflectivity_values, color_values, background_color, bias):
    ray_origins = ray_origins + hit_normals * bias
    ray_directions = ray_directions - 2 * np.einsum('ij,ij->i', ray_directions, hit_normals)[:, np.newaxis] * hit_normals
    hit_colors = reflectivity_values[:, np.newaxis] * color_values + (1 - reflectivity_values[:, np.newaxis]) * background_color
    return ray_origins, ray_directions, hit_colors

# TODO: implement soon
def refract(incident_rays, normals, ior_values):
    cosi = np.clip(np.einsum('ij,ij->i', incident_rays, normals), -1, 1)
    etai = np.ones_like(cosi)
    etat = ior_values
    n = np.where(cosi[:, np.newaxis] < 0, normals, -normals)
    cosi = np.abs(cosi)
    eta = etai / etat
    k = 1 - eta**2 * (1 - cosi**2)
    return np.where(k[:, np.newaxis] < 0, 0, eta[:, np.newaxis] * incident_rays + (eta * cosi - np.sqrt(k))[:, np.newaxis] * n)

def filter_rays(objects, origins, rays):
    in_shadow = np.full(origins.shape[0], False, dtype=bool)
    for object in objects:
        for i in range(origins.shape[0]):
            intersection = object.bvh.search_collision_closest(origins[i], rays[i])
            if intersection is not None:
                in_shadow[i] = True
    return in_shadow

def calculate_scene(w, h, cam, objects, lights):
    background_color = [6, 20, 77]
    image_hit_colors = np.zeros((h * w, 3), dtype=np.float32)
    intersection_mask = None
    max_depth = 3

    primary_rays = cam.primary_rays(w, h)
    ray_origins, ray_directions = cam.position, primary_rays

    for current_depth in range(max_depth):
        print(f'pass {current_depth+1}...')

        st = time.time()
        min_t_values, object_indices, hit_normals, color_values, reflectivity_values, ior_values = trace(objects, ray_origins, ray_directions, background_color)
        print('primary rays cast:', time.time() - st)
        
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
        color_values = color_values[current_intersection_mask]
        reflectivity_values = reflectivity_values[current_intersection_mask]
        ior_values = ior_values[current_intersection_mask]
        object_indices = object_indices[current_intersection_mask]

        filtered_shadow_rays = filter_rays(objects, ray_origins, hit_normals)
        print("Total Rays:", hit_normals.shape[0], "Filtered Rays:", np.sum(filtered_shadow_rays))

        st = time.time()
        hit_colors = shade(objects, lights, ray_origins[filtered_shadow_rays], hit_normals[filtered_shadow_rays], object_indices, background_color)
        
        all_colors = np.full((h * w, 3), [0, 0, 0], dtype=np.float32)
        intersection_indices = np.arange(h * w)[intersection_mask]
        filtered_indices = intersection_indices[filtered_shadow_rays]
        all_colors[filtered_indices] = hit_colors
        image_hit_colors += all_colors

        # image_hit_colors[intersection_mask] += hit_colors
        print('shade compute time:', time.time() - st)

        st = time.time()
        ray_origins, ray_directions, hit_colors = reflect(ray_origins, ray_directions, hit_normals, reflectivity_values, color_values, background_color, 1e-4)
        image_hit_colors[intersection_mask] += hit_colors
        print('reflect compute time:', time.time() - st)

    no_hit_mask = np.all(image_hit_colors == 0, axis=-1)    
    image_hit_colors[no_hit_mask] += background_color

    image_hit_colors = np.clip(image_hit_colors, 0, 255).astype(np.uint8)
    return image_hit_colors

def render_experimental(w, h, cam, objects, lights):
    tst = time.time()
    hit_color_image = calculate_scene(w, h, cam, objects, lights)
    print(f'\nTotal Render Time: {time.time() - tst:.4f}s')

    rendered_image = Image.fromarray(hit_color_image.reshape(h, w, 3), 'RGB')
    rendered_image.show()





# def trace(objects, meshlet_indices, ray_origin, ray_directions):
#     for enum_index, meshlet in enumerate([objects[i] for i in meshlet_indices]):
#         triangles_vertices = meshlet.vertices[meshlet.faces]
#         hit, intersection_points = ray_triangle_intersection(ray_origin, ray_directions, triangles_vertices)

# # do this before each shadow function call to cull useless rays
# def filter_rays(cam, bvh, rays):
#     return np.array([i if bvh.search_collision_all(cam.position, ray) else -1 for i, ray in enumerate(rays)])

# def render_experimental(w, h, cam, bvh, objects):
#     st = time.time()
#     primary_rays = cam.primary_rays(w, h)
#     print('Generate Primary Rays:', time.time() - st)
    
#     st = time.time()
#     primary_mask = filter_rays(cam, bvh, primary_rays)
#     mask = np.where(primary_mask != -1)
#     filtered_rays = primary_rays[mask]
#     print('Cull Rays:', time.time() - st)
#     print(len(primary_rays), len(primary_rays[mask]))

#     st = time.time()
#     trace(objects, np.array([0, 3]), cam.position, primary_rays[5261])
#     print('Primary Ray Cast:', time.time() - st)   

#     # min_t_values, object_indices, _ = trace(objects, cam.position, filtered_rays)
#     # object_i = np.full(primary_rays.shape[0], -1, dtype=int)
#     # object_i[mask] = object_indices

#     # img = np.full((h, w), 127, dtype=np.uint8)
#     # for i, ray in enumerate(primary_rays):
#     #     row, col = i // w, i % w
#     #     img[row, col] = 255 if object_i[i] != -1 else 127

#     # rendered_image = Image.fromarray(img, 'L')
#     # rendered_image.show()

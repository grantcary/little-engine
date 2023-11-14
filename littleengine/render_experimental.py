import sys, time
import numpy as np
from PIL import Image
from numba import njit

np.set_printoptions(threshold=sys.maxsize)

@njit
def optimized_ray_triangle_intersection(origin, direction, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = np.abs(a) < epsilon

    large_number = 1e9
    a_safe = np.where(parallel_mask, large_number, a)

    f = 1.0 / a_safe
    s = origin - v0
    u = f * np.dot(s, h.T)
    valid_u = (u >= 0) & (u <= 1)
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q.T)
    valid_v = (v >= 0) & (u + v <= 1)
    t = f * np.dot(edge2, q.T)
    valid_t = t > epsilon

    intersection_mask = ~parallel_mask & valid_u & valid_v & valid_t
    intersection_points = origin + direction * t
    return intersection_mask, intersection_points

@njit
def numba_optimized_trace(triangles, origins, directions, hit, intersects):
    for i in range(len(directions)):
        for j in range(len(triangles)):
            hit[j, i], intersects[j, i] = optimized_ray_triangle_intersection(origins[i], directions[i], triangles[j])

def optimized_trace(objects, origins, directions, skybox):
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    color = np.zeros((n, 3), dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)
    alpha = np.full(n, 0.0, dtype=np.float64)
    for obj_index, obj in enumerate(objects):
        triangles = obj.vertices[obj.faces]
        m = triangles.shape[0]
        hit = np.zeros((m, n), dtype=bool)
        intersects = np.full((m, n, 3), np.inf, dtype=np.float64)
        numba_optimized_trace(triangles, origins, directions, hit, intersects)

        for i in range(m):
            t_update = np.linalg.norm(intersects[i] - origins, axis=-1)
            mask = (t_update < t) &  hit[i]
            t[mask], obj_indices[mask], normals[mask] = t_update[mask], obj_index, obj.normals[i]
            color[mask], reflectivity[mask], ior[mask], alpha[mask] = obj.color, obj.reflectivity, obj.ior, obj.alpha
    return t, obj_indices, normals, color, reflectivity, ior, alpha

def ray_triangle_intersection(origins, directions, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(directions, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = (np.abs(a) < epsilon)
    
    f = np.zeros_like(a)
    non_zero_a_indices = np.abs(a) >= epsilon
    f[non_zero_a_indices] = 1.0 / a[non_zero_a_indices]

    s = origins - v0
    u = f * np.dot(s, h.T)
    valid_u = (u >= 0) & (u <= 1)
    q = np.cross(s, edge1)
    v = f * np.dot(directions, q.T)
    valid_v = (v >= 0) & (u + v <= 1)
    t = f * np.dot(edge2, q.T)
    valid_t = t > epsilon

    mask = ~parallel_mask & valid_u & valid_v & valid_t
    intersects = origins + directions * t.reshape(-1, 1)
    return mask, intersects

def filter_rays(objects, origins, rays):
    n = origins.shape[0]
    hit = np.zeros(n, dtype=bool)
    for object in objects:
        for i in range(n):
            intersection = object.bvh.search_collision_closest(origins[i], rays[i])
            if intersection is not None:
                hit[i] = True
    return hit

def trace(objects, origins, directions, skybox, use_bvh):
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    colors = np.zeros((n, 3), dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)
    alpha = np.full(n, 0.0, dtype=np.float64)

    use_filter = len(origins.shape) > 1 and use_bvh
    if use_filter: filter_mask = filter_rays(objects, origins, directions)
    origins_filtered = origins[filter_mask] if use_filter else origins
    directions_filtered = directions[filter_mask] if use_filter else directions
    t_filtered = t[filter_mask] if use_filter else t

    for obj_index, obj in enumerate(objects):
        triangles = obj.vertices[obj.faces]
        for tri_index, tri in enumerate(triangles):
            hit, intersects = ray_triangle_intersection(origins_filtered, directions_filtered, tri)
            t_update = np.linalg.norm(intersects - origins_filtered, axis=-1)

            mask = (t_update < t_filtered) & (np.diagonal(hit) if len(hit.shape) > 1 else hit)
            indices = np.where(filter_mask)[0] if use_filter else np.arange(n)

            t[indices[mask]], obj_indices[indices[mask]], normals[indices[mask]] = t_update[mask], obj_index, obj.normals[tri_index]
            colors[indices[mask]], reflectivity[indices[mask]], ior[indices[mask]], alpha[indices[mask]] = obj.color, obj.reflectivity, obj.ior, obj.alpha
    return t, obj_indices, normals, colors, reflectivity, ior, alpha

def shade(objects, lights, intersects, normals, obj_indices, skybox, use_bvh):
    n = intersects.shape[0]
    colors = np.zeros((n, 3), dtype=np.float64)
    t = np.full(n, np.inf, dtype=np.float64)
    indices = np.full(n, -1, dtype=np.int64)

    for light in lights:
        light_directions = light.position - intersects
        len2 = np.sum(light_directions**2, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        t, indices = optimized_trace(objects, intersects, normalized_light_directions, skybox)[:2]

        in_shadow = (indices != -1) & (t**2 < len2)
        cos_theta = np.einsum('ij,ij->i', normals, normalized_light_directions)

        for i in range(n): colors[i] = objects[obj_indices[i]].color * light.intensity * max(0, cos_theta[i]) * (1 - in_shadow[i])
    return colors

def reflect(origins, directions, normals, bias):
    origins = origins + normals * bias
    directions = directions - 2 * np.einsum('ij,ij->i', directions, normals)[:, np.newaxis] * normals
    return origins, directions

def refract(incident_rays, normals, ior_values):
    cosi = np.clip(np.einsum('ij,ij->i', incident_rays, normals), -1, 1)
    etai = np.ones_like(cosi)
    etat = np.where(np.abs(ior_values) < 1e-10, 1e-10, ior_values)
    n = np.where(cosi[:, np.newaxis] < 0, normals, -normals)
    cosi = np.abs(cosi)
    eta = etai / etat
    k = 1 - eta**2 * (1 - cosi**2)
    total_internal_reflection = k[:, np.newaxis] < 0
    reflected = incident_rays - 2 * np.einsum('ij,ij->i', incident_rays, n)[:, np.newaxis] * n
    sqrt_k = np.sqrt(np.maximum(0, k))
    refracted_directions = np.where(total_internal_reflection, reflected, eta[:, np.newaxis] * incident_rays + (eta * cosi - sqrt_k)[:, np.newaxis] * n)
    return np.nan_to_num(refracted_directions)

class DepthNode:
    def __init__(self, bnum, depth, shape):
        self.bnum = bnum
        self.depth = depth
        self.mask = np.zeros(shape, dtype=bool)
        self.shadow = np.zeros((shape, 3), dtype=np.float64)
        self.skybox = np.zeros((shape, 3), dtype=np.float64)
        self.reflectivity = np.ones(shape, dtype=np.float64)
        self.alpha = np.zeros(shape, dtype=np.float64)

        self.main = None
        self.refract = None

    def set_main(self, bnum, depth, shape):
        self.main = DepthNode(bnum, depth, shape)
        return self.main
    
    def set_refract(self, bnum, depth, shape):
        self.refract = DepthNode(bnum, depth, shape)
        return self.refract

def tree_compositor(node, accumulated_reflectivity=None):
    if node.main == None and node.refract == None:
        return np.zeros_like(node.skybox)
    
    output = np.zeros_like(node.shadow)
    if accumulated_reflectivity is None:
        output[node.mask] += node.skybox[node.mask]
        accumulated_reflectivity = np.ones_like(node.reflectivity.reshape(-1, 1))

    reflection_contribution = node.reflectivity[~node.mask, np.newaxis]
    n = accumulated_reflectivity.copy()
    n[~node.mask] *= reflection_contribution

    if node.refract:
        reflect = tree_compositor(node.main, n)
        refract = tree_compositor(node.refract)
        diffuse = node.shadow[~node.mask]
        skybox = node.main.skybox[~node.mask]
        rc, inv_rc = reflection_contribution, (1 - reflection_contribution)
        accum = accumulated_reflectivity[~node.mask]
        alpha = node.alpha[~node.mask, np.newaxis]
        inv_alpha = (1 - alpha)

        r = refract * inv_alpha
        d = (diffuse * inv_rc * accum) * alpha
        s = skybox * rc * accum
        output[~node.mask] += r + d + s

        output += reflect
    else:
        output += tree_compositor(node.main, n)
        output[~node.mask] += node.shadow[~node.mask] * (1 - reflection_contribution) * accumulated_reflectivity[~node.mask]
        output[~node.mask] += node.main.skybox[~node.mask] * reflection_contribution * accumulated_reflectivity[~node.mask]

    return output

def render_thread(branch_number, shape, depth, origins, directions, skybox, objects, lights, use_bvh):
    head, prev = DepthNode(branch_number, 0, shape), None
    for i in range(depth):
        node = head if i == 0 else prev.set_main(branch_number, i, shape)
        if len(origins.shape) > 1:
            t, obj_indices, normals, colors, reflectivity, ior, alpha = optimized_trace(objects, origins, directions, skybox)
        else:
            t, obj_indices, normals, colors, reflectivity, ior, alpha = trace(objects, origins, directions, skybox, use_bvh)

        if len(t) == 0: return head
        intersects = origins + directions * t.reshape(-1, 1)

        current_mask = obj_indices != -1
        if i > 0:
            neg_mask = mask.copy()
            neg_mask[mask] = ~current_mask
            mask[mask] = current_mask
        else:
            mask = current_mask
            neg_mask = ~current_mask

        print(''.join([' '] * branch_number), branch_number, i, len(directions), sum(current_mask), sum(~current_mask))
        if sum(current_mask) == 0 or sum(~current_mask) == 0: return head
        if directions[~current_mask].min() != 0 or directions[~current_mask].max() != 0:
            node.skybox[neg_mask] = skybox.get_texture(directions[~current_mask])

        origins, directions, normals = intersects[current_mask], directions[current_mask], normals[current_mask]
        colors, reflectivity, ior, alpha = colors[current_mask], reflectivity[current_mask], ior[current_mask], alpha[current_mask]

        if depth - i - 1 != 0 and np.any(ior > 1.0) and np.any(ior <= 2.42) and len(directions) != 0:
            refracted = refract(directions, normals, ior) if i % 2 == 0 else refract(directions, -normals, np.where(ior == 0, 1e-10, ior))
            rorigins = origins + refracted * 1e-6 if i % 2 == 0 else origins
            node.refract = render_thread(branch_number + 1, len(directions), depth - i - 1, rorigins, refracted, skybox, objects, lights, use_bvh)
        node.alpha[mask] = alpha

        node.shadow[mask] = shade(objects, lights, origins, normals, obj_indices[current_mask], skybox, use_bvh)
        node.mask = ~mask

        node.reflectivity[mask] = reflectivity
        origins, directions = reflect(origins, directions, normals, 1e-4)
        
        prev = node
    return head

def render_experimental(params, cam, skybox, objects, lights):
    st = time.time()
    shape = params.h * params.w
    origins, directions = cam.position, cam.primary_rays(params.w, params.h)
    tree = render_thread(0, shape, params.depth, origins, directions, skybox, objects, lights, params.use_bvh)

    image = tree_compositor(tree)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image.reshape(params.h, params.w, 3), 'RGB')
    
    render_time = time.time() - st
    return image, render_time
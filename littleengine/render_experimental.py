import sys, time
import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import njit

from typing import List, Tuple
from numpy.typing import NDArray

from littleengine import Camera, Object, Light
from .utils import SceenParams

# np.set_printoptions(threshold=sys.maxsize)

ndf64 = NDArray[np.float64]
ndi8 = NDArray[np.int8]

@njit
def optimized_ray_triangle_intersection(origin, direction, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = np.abs(a) < epsilon

    f = 1.0 / a
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

def optimized_trace(objects, origins, directions, bgc):
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    color = np.full((n, 3), bgc, dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)
    for obj_index, obj in enumerate(objects):
        triangles = obj.vertices[obj.faces]
        m = triangles.shape[0]
        hit = np.zeros((m, n), dtype=bool)
        intersects = np.full((m, n, 3), np.inf, dtype=np.float64)
        numba_optimized_trace(triangles, origins, directions, hit, intersects)

        for i in range(m):
            t_update = np.linalg.norm(intersects[i] - origins, axis=-1)
            mask = (t_update < t) &  hit[i]
            t[mask] = t_update[mask]
            obj_indices[mask] = obj_index
            normals[mask] = obj.normals[i]
            color[mask] = obj.color
            reflectivity[mask] = obj.reflectivity
            ior[mask] = obj.ior
    return t, obj_indices, normals, color, reflectivity, ior

def ray_triangle_intersection(origins: ndf64, directions: ndf64, triangle: ndf64) -> Tuple[ndi8, ndf64]:
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

def filter_rays(objects: List[Object], origins: ndf64, rays: ndf64) -> ndi8:
    n = origins.shape[0]
    hit = np.zeros(n, dtype=bool)
    for object in objects:
        for i in range(n):
            intersection = object.bvh.search_collision_closest(origins[i], rays[i])
            if intersection is not None:
                hit[i] = True
    return hit

def trace(objects: List[Object], origins: ndf64, directions: ndf64, bgc: List[int], use_bvh: bool) -> Tuple[ndf64, ndi8, ndf64, ndf64, ndf64, ndf64]:
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    colors = np.full((n, 3), bgc, dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)

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
            colors[indices[mask]], reflectivity[indices[mask]], ior[indices[mask]] = obj.color, obj.reflectivity, obj.ior
    return t, obj_indices, normals, colors, reflectivity, ior

def shade(objects: List[Object], lights: List[Light], intersects: ndf64, normals: ndf64, obj_indices: ndi8, bgc: ndf64, use_bvh: bool) -> ndf64:
    n = intersects.shape[0]
    colors = np.zeros((n, 3), dtype=np.float64)
    t = np.full(n, np.inf, dtype=np.float64)
    indices = np.full(n, -1, dtype=np.int64)

    for light in lights:
        light_directions = light.position - intersects
        len2 = np.sum(light_directions**2, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        t, indices = optimized_trace(objects, intersects, normalized_light_directions, bgc)[:2]

        in_shadow = (indices != -1) & (t**2 < len2)
        cos_theta = np.einsum('ij,ij->i', normals, normalized_light_directions)

        for i in range(n): colors[i] = objects[obj_indices[i]].color * light.intensity * max(0, cos_theta[i]) * (1 - in_shadow[i])
    return colors

def reflect(origins: ndf64, directions: ndf64, normals: ndf64, colors: ndf64, reflectivity: ndf64, bgc: List[int], bias: float) -> Tuple[ndf64, ndf64, ndf64]:
    origins = origins + normals * bias
    directions = directions - 2 * np.einsum('ij,ij->i', directions, normals)[:, np.newaxis] * normals
    colors = reflectivity[:, np.newaxis] * colors + (1 - reflectivity[:, np.newaxis]) * bgc
    return origins, directions, colors

def render_experimental(cam: Camera, params: SceenParams, objects: List[Object], lights: List[Light]) -> None:
    st = time.time()
    image = np.zeros((params.h * params.w, 3), dtype=np.uint8)
    origins, directions = cam.position, cam.primary_rays(params.w, params.h)
    mask = None
    pbar = tqdm(total=params.depth, desc='Ray Depth')
    for i in range(params.depth):
        if len(origins.shape) > 1:
            t, obj_indices, normals, colors, reflectivity, ior = optimized_trace(objects, origins, directions, params.bgc)
        else:
            t, obj_indices, normals, colors, reflectivity, ior = trace(objects, origins, directions, params.bgc, params.use_bvh)

        if t.shape[0] == 0: pbar.update(params.depth - i); break
        intersects = origins + directions * t.reshape(-1, 1)
        
        current_mask = obj_indices != -1
        if mask is not None: mask[mask] = current_mask
        else: mask = current_mask
        
        origins, directions, normals = intersects[current_mask], directions[current_mask], normals[current_mask]
        colors, reflectivity, ior = colors[current_mask], reflectivity[current_mask], ior[current_mask]
 
        image[mask] += shade(objects, lights, origins, normals, obj_indices[current_mask], params.bgc, params.use_bvh)
        origins, directions, colors = reflect(origins, directions, normals, colors, reflectivity, params.bgc, 1e-4)
        image[mask] += colors

        # infinity = image == np.inf

        pbar.update()
    pbar.close()
   
    image[np.all(image == 0, axis=-1)] += params.bgc
    rendered_image = np.clip(image, 0, 255).astype(np.uint8)

    image = Image.fromarray(rendered_image.reshape(params.h, params.w, 3), 'RGB')
    render_time = time.time() - st
    return image, render_time
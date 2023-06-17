import sys, time

from typing import Type, List, Tuple
from numpy.typing import NDArray

import numpy as np
from PIL import Image
from tqdm import tqdm

from littleengine import *
from .utils import SceenParams

# np.set_printoptions(threshold=sys.maxsize)

ndf64 = NDArray[np.float64]
ndi8 = NDArray[np.int8]

def ray_triangle_intersection(origins: ndf64, directions: ndf64, triangle: ndf64) -> Tuple[ndi8, ndf64]:
    epsilon = 1e-6
    v0, v1, v2 = triangle

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(directions, edge2)
    a = np.dot(edge1, h.T)

    parallel_mask = np.abs(a) < epsilon

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

def trace(objects: List[Object], origins: ndf64, directions: ndf64, bgc: List[int]) -> Tuple[ndf64, ndi8, ndf64, ndf64, ndf64, ndf64]:
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    colors = np.full((n, 3), bgc, dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)

    for obj_index, obj in enumerate(objects):
        triangles = obj.vertices[obj.faces]
        for tri_index, tri in enumerate(triangles):
            hit, intersects = ray_triangle_intersection(origins, directions, tri)
            t_update = np.linalg.norm(intersects - origins, axis=-1)
            hit = np.diagonal(hit) if len(hit.shape) > 1 else hit
            mask = (t_update < t) & hit
            t[mask], obj_indices[mask], normals[mask] = t_update[mask], obj_index, obj.normals[tri_index]
            colors[mask], reflectivity[mask], ior[mask] = obj.color, obj.reflectivity, obj.ior

    return t, obj_indices, normals, colors, reflectivity, ior

def shade(objects: List[Object], lights: List[Light], intersects: ndf64, normals: ndf64, obj_indices: ndi8, bgc: ndf64) -> ndf64:
    n = intersects.shape[0]
    colors = np.zeros((n, 3), dtype=np.float64)

    for light in lights:
        light_directions = light.position - intersects
        len2 = np.sum(light_directions ** 2, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        t, indices = trace(objects, intersects, normalized_light_directions, bgc)[:2]
        in_shadow = (indices != -1) & (t ** 2 < len2)
        cos_theta = np.einsum('ij,ij->i', normals, normalized_light_directions)

        for i in range(n): 
            colors[i] = objects[obj_indices[i]].color * light.intensity * max(0, cos_theta[i]) * (1 - in_shadow[i])

    return colors

def reflect(origins: ndf64, directions: ndf64, normals: ndf64, colors: ndf64, reflectivity: ndf64, bgc: List[int], bias: float) -> Tuple[ndf64, ndf64, ndf64]:
    origins = origins + normals * bias
    directions = directions - 2 * np.einsum('ij,ij->i', directions, normals)[:, np.newaxis] * normals
    colors = reflectivity[:, np.newaxis] * colors + (1 - reflectivity[:, np.newaxis]) * bgc
    return origins, directions, colors

def calculate_scene(w: int, h: int, bgc: List[int], depth: int, cam: Camera, objects: List[Object], lights: List[Light]) -> ndf64:
    image = np.zeros((h * w, 3), dtype=np.float64)
    origins, directions = cam.position, cam.primary_rays(w, h)
    mask = None
    for _ in tqdm(range(depth), total=depth, desc='Ray Depth'):
        t, obj_indices, normals, colors, reflectivity, ior = trace(objects, origins, directions, bgc)
        intersects = origins + directions * t.reshape(-1, 1)
        
        current_mask = obj_indices != -1
        if mask is not None:
            mask[mask] = current_mask
        else:
            mask = current_mask
        
        obj_indices = obj_indices[current_mask]
        origins, directions, normals = intersects[current_mask], directions[current_mask], normals[current_mask]
        colors, reflectivity, ior = colors[current_mask], reflectivity[current_mask], ior[current_mask]
 
        image[mask] += shade(objects, lights, origins, normals, obj_indices, bgc)
        origins, directions, colors = reflect(origins, directions, normals, colors, reflectivity, bgc, 1e-4)
        image[mask] += colors

    no_hit_mask = np.all(image == 0, axis=-1)    
    image[no_hit_mask] += bgc
    return np.clip(image, 0, 255).astype(np.uint8)

def render(cam: Camera, params: SceenParams, objects: List[Object], lights: List[Light]) -> None:
    st = time.time()
    rendered_image = calculate_scene(params.w, params.h, params.bgc, params.depth, cam, objects, lights)
    print(f'\nTotal Render Time: {time.time() - st:.4f}s')

    image = Image.fromarray(rendered_image.reshape(params.h, params.w, 3), 'RGB')
    image.show()
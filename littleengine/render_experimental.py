import sys, time
import numpy as np
from PIL import Image
from tqdm import tqdm

from typing import List, Tuple
from numpy.typing import NDArray

from littleengine import Camera, Object, Light
from .utils import SceenParams

# np.set_printoptions(threshold=sys.maxsize)

ndf64 = NDArray[np.float64]
ndi8 = NDArray[np.int8]

def ray_triangle_intersection(origins: ndf64, directions: ndf64, triangle: ndf64) -> Tuple[ndi8, ndf64]:
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(directions, edge2)
    a = np.dot(edge1, h.T)

    f = np.zeros_like(a)
    non_zero_a_indices = np.abs(a) >= epsilon
    f[non_zero_a_indices] = 1.0 / a[non_zero_a_indices]
    
    s = origins - v0
    q = np.cross(s, edge1)
    t = f * np.dot(edge2, q.T)
    u = f * np.dot(s, h.T)
    v = f * np.dot(directions, q.T)

    mask = ~(np.abs(a) < epsilon) & ((u >= 0) & (u <= 1)) & ((v >= 0) & (u + v <= 1)) & (t > epsilon)
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

def trace(objects: List[Object], origins: ndf64, directions: ndf64, bgc: List[int]) -> Tuple[ndf64, ndi8, ndf64, ndf64, ndf64, ndf64]:
    n = directions.shape[0]
    t = np.full(n, np.inf, dtype=np.float64)
    obj_indices = np.full(n, -1, dtype=np.int8)
    normals = np.full((n, 3), [0.0, 0.0, 0.0], dtype=np.float64)
    colors = np.full((n, 3), bgc, dtype=np.float64)
    reflectivity = np.full(n, 0.0, dtype=np.float64)
    ior = np.full(n, 0.0, dtype=np.float64)

    filter_mask = np.ones(n, dtype=bool) if len(origins.shape) == 1 else filter_rays(objects, origins, directions)
    for obj_index, obj in enumerate(objects):
        triangles = obj.vertices[obj.faces]
        for tri_index, tri in enumerate(triangles):
            if len(origins.shape) > 1:
                hit, intersects = ray_triangle_intersection(origins[filter_mask], directions[filter_mask], tri)
                t_update = np.linalg.norm(intersects - origins[filter_mask], axis=-1)
                mask = (t_update < t[filter_mask]) & np.diagonal(hit)
                t[filter_mask][mask], obj_indices[filter_mask][mask], normals[filter_mask][mask] = t_update[mask], obj_index, obj.normals[tri_index]
                colors[filter_mask][mask], reflectivity[filter_mask][mask], ior[filter_mask][mask] = obj.color, obj.reflectivity, obj.ior
            else:
                hit, intersects = ray_triangle_intersection(origins, directions, tri)
                t_update = np.linalg.norm(intersects - origins, axis=-1)
                mask = (t_update < t) & hit
                t[mask], obj_indices[mask], normals[mask] = t_update[mask], obj_index, obj.normals[tri_index]
                colors[mask], reflectivity[mask], ior[mask] = obj.color, obj.reflectivity, obj.ior
    return t, obj_indices, normals, colors, reflectivity, ior

def shade(objects: List[Object], lights: List[Light], intersects: ndf64, normals: ndf64, obj_indices: ndi8, bgc: ndf64) -> ndf64:
    n = intersects.shape[0]
    colors = np.zeros((n, 3), dtype=np.float64)
    t = np.full(n, np.inf, dtype=np.float64)
    indices = np.full(n, -1, dtype=np.int64)

    for light in lights:
        light_directions = light.position - intersects
        len2 = np.sum(light_directions**2, axis=-1)
        normalized_light_directions = light_directions / np.sqrt(len2).reshape(-1, 1)

        t, indices = trace(objects, intersects, normalized_light_directions, bgc)[:2]
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
    image = np.zeros((params.h * params.w, 3), dtype=np.float64)
    origins, directions = cam.position, cam.primary_rays(params.w, params.h)
    mask = None
    pbar = tqdm(total=params.depth, desc='Ray Depth')
    for i in range(params.depth):
        t, obj_indices, normals, colors, reflectivity, ior = trace(objects, origins, directions, params.bgc)
        if t.shape[0] == 0: pbar.update(params.depth - i); break
        intersects = origins + directions * t.reshape(-1, 1)
        
        current_mask = obj_indices != -1
        if mask is not None: mask[mask] = current_mask
        else: mask = current_mask
        
        origins, directions, normals = intersects[current_mask], directions[current_mask], normals[current_mask]
        colors, reflectivity, ior = colors[current_mask], reflectivity[current_mask], ior[current_mask]
 
        image[mask] += shade(objects, lights, origins, normals, obj_indices[current_mask], params.bgc)
        origins, directions, colors = reflect(origins, directions, normals, colors, reflectivity, params.bgc, 1e-4)
        image[mask] += colors
        
        pbar.update()
    pbar.close()
   
    image[np.all(image == 0, axis=-1)] += params.bgc # if pixel was not hit, it is the background color
    rendered_image = np.clip(image, 0, 255).astype(np.uint8)
    print(f'Total Render Time: {time.time() - st:.4f}s')

    image = Image.fromarray(rendered_image.reshape(params.h, params.w, 3), 'RGB')
    image.show()
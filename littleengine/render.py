import math
import time
import random

import numpy as np
from PIL import Image
from numba import njit

def rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))

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

def trace(obj, ray_origin, ray_directions):
    total_rays = len(ray_directions)
    int_points = np.zeros(total_rays, dtype=bool)
    triangle_vertices = obj.vertices[obj.faces]

    for triangle in triangle_vertices:
        hit, intersection_points = ray_triangle_intersection(ray_origin, ray_directions, triangle)
        # intersection_points -= ray_origin
        int_points |= hit
    return np.where(int_points)[0]

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

def render(w, h, cam, obj):
    """
    Renders an image of the object using the camera.
    """

    rendered_image = Image.new('L', (w, h), 0)
    pixel_buffer = rendered_image.load()

    st = time.time()
    ray_vectors = camera_rays(w, h, cam)
    rays_traced = trace(obj, cam.position, ray_vectors)
    print(time.time() - st)

    rays_traced = np.unique(rays_traced)

    # setup pixel buffer
    for i in range(w):
        for j in range(h):
            pixel_buffer[i, j] = 127

    for ray in rays_traced:
        row, col = ray // w, ray % w
        if row < h and col < w:
            pixel_buffer[col, row] = 255

    rendered_image.show()
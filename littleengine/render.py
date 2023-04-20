import math
import numpy as np
from PIL import Image
import time

def ray_triangle_intersection(ray_origin, ray_direction, triangle_vertices):
    epsilon = 1e-6
    v0, v1, v2 = triangle_vertices

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return False, None

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return False, None

    t = f * np.dot(edge2, q)

    if t > epsilon:
        intersection_point = ray_origin + ray_direction * t
        return True, intersection_point

    return False, None

def trace(obj, ray_origin, ray_directions):
    trace_test = 0
    total_rays = len(ray_directions)
    int_points = []
    for i, ray_direction in enumerate(ray_directions):
        for j, triangle in enumerate(obj.faces):
            hit, intersection_point = ray_triangle_intersection(ray_origin, ray_direction, obj.vertices[triangle])
            if hit:
                # phit = ray_origin + ray_direction * intersection_point
                # nhit = phit - obj.normals[j]

                # if np.dot(ray_direction, nhit) > 0:
                    # nhit = -nhit
                    # print(phit, nhit)

                int_points.append(i)
        print(f'Trace pass {trace_test} of {total_rays}')
        trace_test += 1
    return int_points

def camera_ray_test(w, h, cam):
    aspect_ratio = w / h
    angle = math.tan(math.radians(cam.fov / 2))
    camera_distance = 1 / angle

    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    xx = (2 * (x_indices + 0.5) / w - 1) * angle * aspect_ratio
    yy = (1 - 2 * (y_indices + 0.5) / h) * angle
    a = np.zeros((*xx.shape, 3))
    a[:, :, 0] = xx
    a[:, :, 1] = yy
    a[:, :, 2] = -camera_distance
    normalized_a = a / np.linalg.norm(a, axis=-1)[:, :, np.newaxis]

    return normalized_a.reshape(-1, 3)

def render(w, h, cam, obj):
    """
    Renders an image of the object using the camera.
    """

    rendered_image = Image.new('L', (w, h), 0)
    pixel_buffer = rendered_image.load()

    ray_vectors = camera_ray_test(w, h, cam).reshape(-1, 3)
    
    st = time.time()
    rays_traced = trace(obj, cam.position, ray_vectors)
    print(time.time() - st)

    rays_traced = np.unique(rays_traced)

    for i in range(w):
        for j in range(h):
            pixel_buffer[i, j] = 127

    for ray in rays_traced:
        row, col = ray // w, ray % h
        if row < w and col < h:
            pixel_buffer[row, col] = 255

    rendered_image.show()
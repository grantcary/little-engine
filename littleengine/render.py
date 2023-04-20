import math
import numpy as np
from PIL import Image

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
    int_points = []
    for ray_direction in ray_directions:
        for triangle in obj.faces:
            hit, intersection_point = ray_triangle_intersection(ray_origin, ray_direction, obj.vertices[triangle])
            # print(hit, intersection_point)
            if hit:
                int_points.append(triangle)
    return np.array(int_points)

def camera_ray_test(cam):
    camera_distance = 1 / math.tan(math.radians(cam.fov / 2))
    ph = math.tan(math.radians(cam.fov / 2))
    pw = ph * cam.aspect_ratio
    angle = ph
    h, w = 10, 10
    ih, iw = 1/h, 1/w
    pxarray = []
    for y in range(h):
        for x in range(w):
            xx = (2 * (x + 0.5) * iw - 1) * angle * cam.aspect_ratio
            yy = (1 - 2 * ((y + 0.5) * ih)) * angle
            a = np.array([xx, yy, -camera_distance])
            pxarray.append(a / np.linalg.norm(a))

    return np.array(pxarray)

def test_trace():
    pass

def render(w, h, cam, obj):
    """
    Renders an image of the object using the camera.
    """
    ren = Image.new('L', (w, h), 0)
    pix = ren.load()
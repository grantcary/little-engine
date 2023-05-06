import numpy as np
from numba import njit

import littleengine.mesh as mesh

class BVH():
    def __init__(self, object_indices, triangle_indices, left = None, right = None):
        self.object_indices   : np.array[np.array[int]] = object_indices
        self.triangle_indices : np.array[np.array[int]] = triangle_indices
        self.left  : BVH = left
        self.right : BVH = right

class Bounding_Box():
    def __init__(self, bounds):
        self.bounds = bounds

    def intersect(self, ray_origin, ray_direction):
        invdir = 1 / ray_direction
        flipped_axes = np.argwhere(invdir < 0).flatten()
        self.bounds[flipped_axes] = self.bounds[flipped_axes, ::-1]

        mm = (self.bounds - ray_origin.reshape(3, 1)) * invdir.reshape(3, 1)
        tmin, tmax, tymin, tymax, tzmin, tzmax = mm.flatten()

        if (tmin > tymax) or (tymin > tmax):
            return None
        
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        if (tmin > tzmax) or (tzmin > tmax):
            return None
        
        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax

        t = tmin

        if t < 0:
            t = tmax
            if t < 0:
                return None
        
        return t

def bounds_generator(vertices):
    return np.vstack([np.array([vertices.max(axis=0), vertices.min(axis=0)]).T])
    # dimensions = np.abs(xyz_mm[:, 1] - xyz_mm[:, 0])
    # center = (xyz_mm[:, 1] + xyz_mm[:, 0]) / 2

def bounding_volume_hierarchy(objects):
    pass
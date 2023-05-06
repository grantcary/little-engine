import numpy as np

import littleengine.mesh as mesh

class BVH():
    def __init__(self, object_indices, triangle_indices, left = None, right = None):
        self.object_indices   : np.array[np.array[int]] = object_indices
        self.triangle_indices : np.array[np.array[int]] = triangle_indices
        self.left  : BVH = left
        self.right : BVH = right

class Bounding_Box():
    def __init__(self, xyz_mm):
        self.xyz_mm = xyz_mm

    def intersect(self):
        pass

def bounding_box_generator(vertices):
    return np.vstack([np.array([vertices.max(axis=0), vertices.min(axis=0)]).T])
    # dimensions = np.abs(xyz_mm[:, 1] - xyz_mm[:, 0])
    # center = (xyz_mm[:, 1] + xyz_mm[:, 0]) / 2

def bounding_volume_hierarchy(objects):
    pass
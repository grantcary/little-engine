import numpy as np

import littleengine.mesh as mesh

class BVH():
    def __init__(self):
        self.left = None
        self.right = None
        pass

class Bounding_Box():
    def __init__(self, height, width, length, center = None):
        self.height = height
        self.width = width
        self.lenght = length
        self.center = center

def bounding_box_generator(object):
    xyz_mm = np.vstack([np.array([object.vertices.max(axis=0), object.vertices.min(axis=0)]).T])
    dimensions = np.abs(xyz_mm[:, 1] - xyz_mm[:, 0])
    center = (xyz_mm[:, 1] + xyz_mm[:, 0]) / 2
    return dimensions, center

def bounding_volume_hierarchy(objects):
    pass
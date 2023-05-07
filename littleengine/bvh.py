import numpy as np

class BVH():
    def __init__(self, object_indices, triangle_indices, left = None, right = None):
        self.object_indices   : np.array[np.array[int]] = object_indices
        self.triangle_indices : np.array[np.array[int]] = triangle_indices
        self.left  : BVH = left
        self.right : BVH = right

class Bounding_Box():
    def __init__(self, vertices):
        self.bounds = np.vstack([np.array([vertices.max(axis=0), vertices.min(axis=0)]).T])

    def intersect(self, ray_origin, ray_direction):
        invdir = 1 / ray_direction
        tx = (self.bounds - ray_origin.reshape(3, 1)) * invdir.reshape(3, 1)
        t1, t2, t3, t4, t5, t6 = tx.flatten()
        
        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

        return None if tmax < 0 or tmin > tmax else tmin

def bvh_generator(scene):
    pass

def bounding_volume_hierarchy(objects):
    pass
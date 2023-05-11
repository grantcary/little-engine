import numpy as np

class BVH():
    def __init__(self, bounding_box, object_index = None, left = None, right = None, leaf = False):
        self.bounding_box = bounding_box
        self.object_index = object_index
        self.left = left
        self.right = right
        self.leaf = leaf

    def search_collision(self, ray_origin, ray_direction):
        intersection = self.bounding_box.intersect(ray_origin, ray_direction)
        if intersection is None:
            return None
        elif self.leaf == True:
            return intersection
        
        return self.left.search_collision(ray_origin, ray_direction) or self.right.search_collision(ray_origin, ray_direction)

class Bounding_Box():
    def __init__(self, vertices = None, bounds = None):
        self.bounds = self.get_bounds(vertices) if isinstance(vertices, np.ndarray) else bounds

    def get_bounds(self, vertices):
        return np.vstack([np.array([vertices.max(axis=0), vertices.min(axis=0)]).T])

    def intersect(self, ray_origin, ray_direction):
        invdir = 1 / ray_direction
        tx = (self.bounds - ray_origin.reshape(3, 1)) * invdir.reshape(3, 1)
        t1, t2, t3, t4, t5, t6 = tx.flatten()
        
        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

        return None if tmax < 0 or tmin > tmax else tmin

def bounding_volume_hierarchy(scene) -> BVH:
    pass
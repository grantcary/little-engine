import numpy as np

class BVH():
    def __init__(self, bounding_box = None, left = None, right = None, leaf = False):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.leaf = leaf

    def get_intersection(self, ray_origin, ray_direction, side):
        return None if side is None else side.bounding_box.intersect(ray_origin, ray_direction)

    def search_collision(self, ray_origin, ray_direction, closest=False):
        intersection = self.bounding_box.intersect(ray_origin, ray_direction)
        if intersection is None:
            return [] if not closest else None
        elif self.leaf:
            return [self.object_index] if not closest else intersection

        left_intersection = self.get_intersection(ray_origin, ray_direction, self.left)
        right_intersection = self.get_intersection(ray_origin, ray_direction, self.right)

        if left_intersection is None and right_intersection is None:
            return [] if not closest else None
        elif left_intersection is None or (right_intersection is not None and left_intersection > right_intersection):
            return self.right.search_collision(ray_origin, ray_direction, closest)
        else:
            return self.left.search_collision(ray_origin, ray_direction, closest)

    def search_collision_all(self, ray_origin, ray_direction):
        return self.search_collision(ray_origin, ray_direction, closest=False)

    def search_collision_closest(self, ray_origin, ray_direction):
        return self.search_collision(ray_origin, ray_direction, closest=True)

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

def merge_bounds(b1, b2):
    return np.column_stack((np.maximum(b1[:, 0], b2[:, 0]), np.minimum(b1[:, 1], b2[:, 1])))

def generate_and_build_tree(object, meshlets):
    if len(meshlets) == 1:
        return BVH(bounding_box=Bounding_Box(object.vertices[object.faces[meshlets[0].triangles]].reshape(-1, 3)), leaf=True)

    mid = len(meshlets) // 2

    left = generate_and_build_tree(object, meshlets[:mid])
    right = generate_and_build_tree(object, meshlets[mid:])

    node = BVH(left=left, right=right)

    bounds_left = left.bounding_box.bounds if left else None
    bounds_right = right.bounding_box.bounds if right else None

    if bounds_left is not None and bounds_right is not None:
        node.bounding_box = Bounding_Box(bounds=merge_bounds(bounds_left, bounds_right))
    elif bounds_left is not None:
        node.bounding_box = left.bounding_box
    elif bounds_right is not None:
        node.bounding_box = right.bounding_box

    return node

def bounding_volume_hierarchy(object, meshlets) -> BVH:
    return generate_and_build_tree(object, meshlets)
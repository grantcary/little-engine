import numpy as np

class BVH():
    def __init__(self, bounding_box = None, object_index = None, left = None, right = None, leaf = False):
        self.bounding_box = bounding_box
        self.object_index = object_index
        self.left = left
        self.right = right
        self.leaf = leaf

    # def search_collision(self, ray_origin, ray_direction):
    #     intersection = self.bounding_box.intersect(ray_origin, ray_direction)
    #     if intersection is None:
    #         return []
    #     elif self.leaf:
    #         return [self.object_index]

    #     indices = []

    #     left_intersection = None if self.left is None else self.left.bounding_box.intersect(ray_origin, ray_direction)
    #     right_intersection = None if self.right is None else self.right.bounding_box.intersect(ray_origin, ray_direction)

    #     if left_intersection is not None:
    #         indices.extend(self.left.search_collision(ray_origin, ray_direction))
    #     if right_intersection is not None:
    #         indices.extend(self.right.search_collision(ray_origin, ray_direction))

    #     return indices

    def search_collision(self, ray_origin, ray_direction):
        intersection = self.bounding_box.intersect(ray_origin, ray_direction)
        if intersection is None:
            return None
        elif self.leaf == True:
            return intersection

        left_intersection = self.left.bounding_box.intersect(ray_origin, ray_direction)
        right_intersection = self.right.bounding_box.intersect(ray_origin, ray_direction)

        if left_intersection is None and right_intersection is None:
            return None

        elif left_intersection is None:
            return self.right.search_collision(ray_origin, ray_direction)
        elif right_intersection is None:
            return self.left.search_collision(ray_origin, ray_direction)

        elif left_intersection < right_intersection:
            left_collision = self.left.search_collision(ray_origin, ray_direction)
            if left_collision is not None and left_collision < right_intersection:
                return left_collision
            return self.right.search_collision(ray_origin, ray_direction) or left_collision

        else:
            right_collision = self.right.search_collision(ray_origin, ray_direction)
            if right_collision is not None and right_collision < left_intersection:
                return right_collision
            return self.left.search_collision(ray_origin, ray_direction) or right_collision

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

def merge_bounds(b1, b2):
    return np.column_stack((np.maximum(b1[:, 0], b2[:, 0]), np.minimum(b1[:, 1], b2[:, 1])))

def gen_meshlet_tree(meshlets):
    if len(meshlets) == 0:
        return None
    
    if len(meshlets) == 1:
        return BVH(object_index=meshlets[0].index, leaf=True)

    mid = len(meshlets) // 2

    left = gen_meshlet_tree(meshlets[:mid])
    right = gen_meshlet_tree(meshlets[mid:])

    return BVH(left=left, right=right)

def build_meshlet_bounds(object, meshlets, node):
    if node is None:
        return None

    if node.left:
        build_meshlet_bounds(object, meshlets, node.left)
    if node.right:
        build_meshlet_bounds(object, meshlets, node.right)

    if node.leaf:
        node.bounding_box = Bounding_Box(object.vertices[object.faces[meshlets[node.object_index].triangles]].reshape(-1, 3))
    elif node.left and node.right:
        node.bounding_box = Bounding_Box(bounds=merge_bounds(node.left.bounding_box.bounds, node.right.bounding_box.bounds))
    elif node.left:
        node.bounding_box = node.left.bounding_box
    elif node.right:
        node.bounding_box = node.right.bounding_box
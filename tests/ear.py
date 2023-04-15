import numpy as np

# verts_2D = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])[::-1, :2]

def is_point_inside_triangle(point, v0, v1, v2):
    b0 = np.cross(v2 - v0, point - v0) >= 0
    b1 = np.cross(v0 - v1, point - v1) >= 0
    b2 = np.cross(v1 - v2, point - v2) >= 0

    return b0 == b1 and b1 == b2

def ear_clipping_triangulation_np(polygon):
    indices = np.arange(len(polygon))
    triangles = []

    while len(indices) > 3:
        for i in range(len(indices)):
            i_prev = (i - 1) % len(indices)
            i_next = (i + 1) % len(indices)
            v0, v1, v2 = polygon[indices[[i_prev, i, i_next]]]
            
            va = v1 - v0
            vb = v2 - v0

            if np.cross(va, vb) <= 0:
                continue

            is_ear = True

            for j in range(len(indices)):
                if j == i_prev or j == i or j == i_next:
                    continue
                point = polygon[indices[j]]
                if is_point_inside_triangle(point, v0, v1, v2):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((indices[i_prev], indices[i], indices[i_next]))
                indices = np.delete(indices, i)
                break

        if len(indices) == 3:
            triangles.append(tuple(indices))

    # assert len(triangles) == len(polygon) - 2
    return triangles
import numpy as np

class MeshOps:
    def translate(self, x, y, z):
        self.vertices += np.array([x, y, z])

    def set_normals(self):
        face = np.take(self.vertices, self.faces, 0)
        AB = face[:, 1] - face[:, 0]
        AC = face[:, 2] - face[:, 0]
        normal = np.cross(AB, AC)
        self.normals = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

    def triangulate(self):
        # Videos explaining the algorithm:
        # https://www.youtube.com/watch?v=QAdfkylpYwc
        # https://www.youtube.com/watch?v=hTJFcHutls8
        # https://github.com/twobitcoder101/Polygon-Triangulation/blob/main/TriangulatePolygon.cs

        def signed_area(polygon):
            area = 0
            for i in range(len(polygon)):
                v1, v2 = polygon[i], polygon[(i + 1) % len(polygon)]
                area += (v1[0] * v2[1] - v1[1] * v2[0])
            return 0.5 * area
        
        def is_point_inside_triangle(point, v0, v1, v2):
            b0 = np.cross(v2 - v0, point - v0) >= 0
            b1 = np.cross(v0 - v1, point - v1) >= 0
            b2 = np.cross(v1 - v2, point - v2) >= 0
            return b0 == b1 and b1 == b2

        def ear_clipping(polygon):
            indices = np.arange(len(polygon))
            triangles = []
            while len(indices) > 3:
                for i in range(len(indices)):
                    i_prev, i_next = (i - 1) % len(indices), (i + 1) % len(indices)             # get previous and next indices
                    v0, v1, v2 = polygon[indices[[i_prev, i, i_next]]]                          # get vertices of the current triangle
                    if np.cross(v1 - v0, v2 - v0) <= 0:                                         # check if angle is obtuse
                        continue

                    is_ear = True
                    for j in range(len(indices)):                                               # find ear loop
                        if j != i_prev or j != i or j != i_next:                                # check if vertex j is not any of the current triangle's vertices
                            if is_point_inside_triangle(polygon[indices[j]], v0, v1, v2):       # check if vertex j is inside the current triangle
                                is_ear = False
                                break

                    if is_ear:
                        triangles.append((indices[i_prev], indices[i], indices[i_next]))        # add current triangle to the list of triangles
                        indices = np.delete(indices, i)                                         # remove current vertex index from the list of available vertices
                        break
                    
                if len(indices) == 3:
                    triangles.append(tuple(indices))                                            # add the last triangle to the list of triangles
          
            return triangles
        
        self.set_normals()

        triangles = []
        for i, f in enumerate(self.faces):
            if len(f) > 3:
                verts = np.delete(self.vertices[f], np.argmax(np.abs(self.normals[i])), axis=1) # take all vertices of a given face and remove its largest axis
                verts = verts[::-1] if signed_area(verts) < 0 else verts                        # check if polygon is clockwise or counter-clockwise
                triangles.extend(f[list(triangle)] for triangle in ear_clipping(verts))         # triangulate polygon and add to list of triangles
            else:
                triangles.append(f)

        self.faces = np.array(triangles)
        self.set_normals()                                                                      # update normals
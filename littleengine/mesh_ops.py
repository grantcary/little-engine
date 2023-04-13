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


def precondition_faces():
    # choose face for clipping
    def not_triangle(self):
        return [i for i in range(len(self.faces)) if len(self.faces[i]) > 3]
    
    # ear clipping algorithm on list before converting to numpy array
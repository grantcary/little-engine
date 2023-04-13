import numpy as np

class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray = None):
        self.vertices : np.array[np.array[float]] = vertices
        self.faces    : np.array[np.array[int]]   = faces     # array of vertex indices
        self.normals  : np.array[np.array[float]] = normals
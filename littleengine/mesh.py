import numpy as np

# These classes are not meant to be used directly for object creation, just containers for data.

class Vertex:
    def __init__(self, position: np.ndarray):
        self.coords = position
        self.faces = []

class Face:
    def __init__(self, vertices: list[Vertex]):
        self.vertices = vertices # expect only 3 vertices
        self.normal = self.set_normal()

class Mesh:
    # only triangle polygons supported
    def __init__(self, vertices: list[Vertex] = None, faces: list[Face] = None) -> None:
        self.vertices = vertices
        self.faces = faces
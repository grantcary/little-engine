from littleengine.mesh import Mesh
from littleengine.read_file import OBJ
import numpy as np

class Object:
    def __init__(self, name: str = None, path: str = None):
        self.name = name
        self.position = np.array([0.0, 0.0, 0.0])
        self.mesh = Mesh(*OBJ().read(path))

        self.mesh.triangulate()
        
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.normals = self.mesh.normals

        self.material_type = None
        self.color = np.array([255, 255, 255])
        self.luma = 0.0

    def translate(self, x, y, z):
        self.position += np.array([x, y, z])
        self.mesh.translate(x, y, z)
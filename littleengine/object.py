from littleengine.mesh import Mesh
from littleengine.read_file import OBJ

class Object:
    # this class should contain mesh and material data
    # DO NOT LET THIS CLASS BECOME BLOATED
    def __init__(self, name: str = None, path: str = None):
        self.name = name
        v, f = OBJ().read(path)
        self.mesh = Mesh(v, f)
        self.mesh.triangulate()
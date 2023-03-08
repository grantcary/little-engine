class Tools:
  def slope(self, origin: list[float], target: list[float]):
    return [target[i] - origin[i] for i in range(len(origin))]

class Point():
  def __init__(self, x: float, y: float, z: float):
    self.xyz: list = [x, y, z]
    
  def translate(self, x: float, y: float, z: float) -> None:
    self.xyz = [pos + [x, y, z][i] for i, pos in enumerate(self.xyz)]

  def __repr__(self):
    return f"Point({self.xyz[0]}, {self.xyz[1]}, {self.xyz[2]})"

class Vertex():
  def __init__(self, x: float, y: float, z: float, faces: list['Face'] = None) -> None:
    self.xyz: list = [x, y, z]
    self.faces: list = faces

  def translate(self, x: float, y: float, z: float) -> None:
    self.xyz = [pos + [x, y, z][i] for i, pos in enumerate(self.xyz)]

  def moveto(self, target: Point) -> None:
    self.xyz = [pos + [target.xyz][i] for i, pos in enumerate(self.xyz)]

  def __repr__(self):
    return f"Vertex({self.xyz[0]}, {self.xyz[1]}, {self.xyz[2]})"

class Face(Tools):
  def __init__(self, *vertices: list[Vertex]) -> None:
    self.vertices = vertices
    self.normal = self.normal()

  # TODO: calculate normal vector
  def normal(self):
    A = self.slope(self.vertices[1].xyz, self.vertices[0].xyz)
    B = self.slope(self.vertices[2].xyz, self.vertices[0].xyz)
    Nx = A[1] * B[2] - A[2] * B[1]
    Ny = A[2] * B[0] - A[0] * B[2]
    Nz = A[0] * B[1] - A[1] * B[0]
    return [Nx, Ny, Nz]

class Object(Tools):
  def __init__(self, vertices: list[Vertex] = None, faces: list[Face] = None) -> None:
    self.vertices: list = vertices
    self.faces: list = faces
    self.origin: Point = self.set_origin()

  def set_origin(self) -> Point:
    points = [p.xyz for p in self.vertices]
    x = (max(points, key=lambda p: p[0])[0] + min(points, key=lambda p: p[0])[0]) / 2
    y = (max(points, key=lambda p: p[1])[1] + min(points, key=lambda p: p[1])[1]) / 2
    z = (max(points, key=lambda p: p[2])[2] + min(points, key=lambda p: p[2])[2]) / 2
    return Point(x, y, z)

  def translate(self, x: float, y: float, z: float) -> None:
    for vertex in self.vertices:
      vertex.translate(x, y, z)
    self.origin = self.origin.translate(x, y, z)

  def moveto(self, target: Point) -> None:
    x, y, z = self.slope(self.origin.xyz, target.xyz)
    for vertex in self.vertices:
      vertex.translate(x, y, z)
    self.origin = target

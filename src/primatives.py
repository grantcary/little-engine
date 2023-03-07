class Point():
  def __init__(self, x: float, y: float, z: float):
    self.x = x
    self.y = y
    self.z = z
    
  def translate(self, x: float, y: float, z: float) -> None:
    self.x += x
    self.y += y
    self.z += z

  def __repr__(self):
    return f"Point({self.x}, {self.y}, {self.z})"

class Vertex():
  def __init__(self, x: float, y: float, z: float):
    self.x = x
    self.y = y
    self.z = z

  def translate(self, x: float, y: float, z: float) -> None:
    self.x += x
    self.y += y
    self.z += z
    
  def moveto(self, target: Point) -> None:
    self.x = target.x
    self.y = target.y
    self.z = target.z

  def __repr__(self):
    return f"Vertex({self.x}, {self.y}, {self.z})"

# linked list point cloud
class Edge():
  def __init__(self, vertex: Vertex, connected: list['Edge'] = None):
    self.vertex = vertex
    self.connected = connected

class Face():
  def __init__(self, edges = []):
    self.edges = edges

  # TODO: create vertices from edges
  # TODO: calculate normal vector

  # face/edge relation rules:
  # you can't have an face on an open edge loop
  # you can't have a face with less than 3 edges
  # you can have an edge loop with no face

class Object():
  def __init__(self, vertices: list[Vertex] = None, edges: Edge = None):
    self.vertices: list[Vertex] = vertices
    self.edges: Edge = edges
    self.origin: Point = self.set_origin() if self.vertices else None

  def set_origin(self) -> Point:
    points = [p for p in self.vertices]
    x = (max(points, key=lambda p: p.x).x + min(points, key=lambda p: p.x).x) / 2
    y = (max(points, key=lambda p: p.y).y + min(points, key=lambda p: p.y).y) / 2
    z = (max(points, key=lambda p: p.z).z + min(points, key=lambda p: p.z).z) / 2
    return Point(x, y, z)

  def translate(self, x: float, y: float, z: float) -> None:
    for vertex in self.vertices:
      vertex.translate(x, y, z)
    self.origin = self.origin.translate(x, y, z)

  def moveto(self, target: Point) -> None:
    x_delta = target.x - self.origin.x
    y_delta = target.y - self.origin.y
    z_delta = target.z - self.origin.z
    for vertex in self.vertices:
      vertex.translate(x_delta, y_delta, z_delta)
    self.origin = target

  def traverse_edges(self) -> list[Point]:
    def t(edge: Edge, found: list[Point]):
      if edge.vertex in found:
        return
      found.append(edge.vertex)
      if edge.connected == None:
        return
      for e in edge.connected:
        t(e, found)
        
    found = []
    t(self.edges, found)
    return found

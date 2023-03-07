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


class Vertex(Point):
  def __init__(self, points: list[Point] = []):
    # points should all be same x, y, z
    self.points = points

  def translate(self, x: float, y: float, z: float) -> None:
    for point in self.points:
      point.translate(x, y, z)
    
  def moveto(self, target: Point) -> None:
    for point in self.points:
      point.x = target.x
      point.y = target.y
      point.z = target.z

# linked list point cloud
class Edge():
  def __init__(self, point: Point, vertices: list['Edge'] = None):
    self.point = point
    self.vertices = vertices

class Face():
  def __init__(self, edges = []):
    self.edges = edges

  # TODO: create vertices from edges
  # TODO: calculate normal vector

class Object():
  def __init__(self, vertices: list[Vertex] = None, edges: Edge = None):
    self.vertices: list[Vertex] = vertices
    self.edges: Edge = edges
    self.origin: Point = self.set_origin() if self.vertices else None

  def set_origin(self) -> Point:
    points = [p.points[0] for p in self.vertices]
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
      if edge.point in found:
        return
      found.append(edge.point)
      if edge.vertices == None:
        return
      for e in edge.vertices:
        t(e, found)

    found = []
    t(self.edges, found)
    print(found)
    return found
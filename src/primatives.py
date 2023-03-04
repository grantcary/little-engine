class Point():
  def __init__(self, x: float, y: float, z: float):
    self.x = x
    self.y = y
    self.z = z
    
  def translate(self, x: float, y: float, z: float):
    self.x += x
    self.y += y
    self.z += z

class Vertex(Point):
  def __init__(self, points: list[Point] = []):
    # points should all be same x, y, z
    self.points = points

  def translate(self, x: float, y: float, z: float):
    for point in self.points:
      point.translate(x, y, z)
    
  def moveto(self, target: Point):
    for point in self.points:
      point.x = target.x
      point.y = target.y
      point.z = target.z

class Edge():
  def __init__(self, points = []):
    self.points = points

# TODO: find origin of object using vertices
# max and min of x, y, z respectively, then calculate midpoint for each axis

class Object():
  def __init__(self, vertices: list[Vertex] = [], edges: list[Edge] = []):
    self.vertices = vertices
    self.edges = edges

  # TODO: this is inconsistent, get average of all vertices
  def find_origin(self) -> Point:
    points = [p.points[0] for p in self.vertices]
    x = (max(points, key=lambda p: p.x).x + min(points, key=lambda p: p.x).x) / 2
    y = (max(points, key=lambda p: p.y).y + min(points, key=lambda p: p.y).y) / 2
    z = (max(points, key=lambda p: p.z).z + min(points, key=lambda p: p.z).z) / 2
    return Point(x, y, z)

#     def translate(self, x: float, y: float, z: float):
#         for vertex in self.vertices:
#             vertex.translate(x, y, z)

#     def moveto(self, target: Point):
#         for vertex in self.vertices:
#             vertex.moveto(target)
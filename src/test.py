from primatives import Point, Vertex, Edge, Object
import random
import copy

def randomized_point_cloud():
  def create_object():
    vertices = []
    for i in range(10):
      p1 = Point(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11))
      p2 = copy.deepcopy(p1)
      p3 = copy.deepcopy(p1)
      vertices.append(Vertex([p1, p2, p3]))
      print(f"{i}: {p1.x}, {p1.y}, {p1.z}")
    return Object(vertices)
  obj = create_object()
  p = obj.origin
  print(f"Origin: {p.x}, {p.y}, {p.z}")

  print("\nMoving object to random point\n")

  obj.moveto(Point(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11)))
  p = obj.origin
  for i in obj.vertices:
    print(f"{i.points[0].x}, {i.points[0].y}, {i.points[0].z}")
  print(f"Origin: {p.x}, {p.y}, {p.z}")

def edge_loop():
  tr = Point(2, 2, 0)
  tl = Point(-2, 2, 0)
  br = Point(2, -2, 0)
  bl = Point(-2, -2, 0)

  tr_e = Edge(tr)
  tl_e = Edge(tl)
  br_e = Edge(br)
  bl_e = Edge(bl)

  tr_e.vertices = tl_e
  tl_e.vertices = bl_e
  bl_e.vertices = br_e
  br_e.vertices = tr_e

  start = tr_e
  for _ in range(8):
    print(start.point)
    start = start.vertices

def complex_edge_mesh():
  pass

edge_loop()
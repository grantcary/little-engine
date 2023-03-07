from primatives import Point, Vertex, Edge, Object
from matplotlib import pyplot as plt
import random
import copy

def randomized_point_cloud():
  def create_object():
    vertices = []
    for _ in range(10):
      vertices.append(Vertex(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11)))
    return Object(vertices)
  obj = create_object()
  p = obj.origin
  print(f"Origin: {p.x}, {p.y}, {p.z}")

  print("\nMoving object to random point\n")

  obj.moveto(Point(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11)))
  p = obj.origin
  for i in obj.vertices:
    print(f"{i.x}, {i.y}, {i.z}")
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
    print(repr(start.point))
    start = start.vertices

def prism_mesh():
  tf = Vertex(0, 2, 2)
  blf = Vertex(-2, 2, 0)
  brf = Vertex(2, 2, 0)
  tb = Vertex(0, -2, 2)
  blb = Vertex(-2, -2, 0)
  brb = Vertex(2, -2, 0)
  leg = Vertex(4, -2, 0)

  tf_e = Edge(tf)
  blf_e = Edge(blf)
  brf_e = Edge(brf)
  tb_e = Edge(tb)
  blb_e = Edge(blb)
  brb_e = Edge(brb)
  leg_e = Edge(leg)

  tf_e.connected = [tb_e, blf_e, brf_e]  
  blf_e.connected = [brf_e, tf_e, blb_e]
  brf_e.connected = [blf_e, tf_e, brb_e]
  tb_e.connected = [tf_e, blb_e, brb_e]
  blb_e.connected = [brb_e, tb_e, blf_e]
  brb_e.connected = [blb_e, tb_e, brf_e, leg_e]

  def random_traversal():
    start = tf_e
    for _ in range(12):
      print(start.vertex)
      rand = random.randint(0, 2)
      start = start.connected[rand]

  vertices = [tf, blf, brf, tb, blb, brb, leg]
  return Object(vertices, tf_e).traverse_edges()

def complex_edge_mesh():
  pass

def plot_points_3D(points: list[Point, Vertex]):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in points:
    ax.scatter(i.x, i.y, i.z, marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

if __name__ == "__main__":
  found = prism_mesh()
  print(found)
  plot_points_3D(found)
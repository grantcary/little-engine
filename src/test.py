from primatives import Point, Vertex, Edge, Object
from matplotlib import pyplot as plt
import numpy as np
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
    print(repr(start.point))
    start = start.vertices

def prism_mesh():
  tf = Point(0, 2, 2)
  blf = Point(-2, 2, 0)
  brf = Point(2, 2, 0)

  tb = Point(0, -2, 2)
  blb = Point(-2, -2, 0)
  brb = Point(2, -2, 0)

  leg = Point(4, -2, 0)

  tf_e = Edge(tf)
  blf_e = Edge(blf)
  brf_e = Edge(brf)

  tb_e = Edge(tb)
  blb_e = Edge(blb)
  brb_e = Edge(brb)

  leg_e = Edge(leg)

  tf_e.vertices = [tb_e, blf_e, brf_e]  
  blf_e.vertices = [brf_e, tf_e, blb_e]
  brf_e.vertices = [blf_e, tf_e, brb_e]
  
  tb_e.vertices = [tf_e, blb_e, brb_e]
  blb_e.vertices = [brb_e, tb_e, blf_e]
  brb_e.vertices = [blb_e, tb_e, brf_e, leg_e]

  def random_traversal():
    start = tf_e
    for _ in range(12):
      print(start.point)
      rand = random.randint(0, 2)
      start = start.vertices[rand]

  return Object(edges=tf_e).traverse_edges()

def complex_edge_mesh():
  pass

def plot_points_3D(found: list[Point]):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in found:
    ax.scatter(i.x, i.y, i.z, marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

found = prism_mesh()
plot_points_3D(found)
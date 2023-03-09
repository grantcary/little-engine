import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.mesh import Point, Vertex, Face, Object
from tools import plot_points_3D

def complex_edge_mesh_generator():
  pass

def square_pyramid_generator(width: float, height: float):
  v0 = Vertex(0, 0, height)
  v1 = Vertex(width, width, 0)
  v2 = Vertex(-width, width, 0)
  v3 = Vertex(-width, -width, 0)
  v4 = Vertex(width, -width, 0)

  f0 = Face(v0, v1, v2)
  f1 = Face(v0, v2, v3)
  f2 = Face(v0, v3, v4)
  f3 = Face(v0, v4, v1)
  f4 = Face(v1, v2, v3, v4)

  v0.faces = [f0, f1, f2, f3]
  v1.faces = [f0, f3, f4]
  v2.faces = [f0, f1, f4]
  v3.faces = [f1, f2, f4]
  v4.faces = [f2, f3, f4]

  return Object([v0, v1, v2, v3, v4], [f0, f1, f2, f3, f4])

if __name__ == "__main__":
  obj = square_pyramid_generator(4, 4)
  obj.moveto(Point(0, 0, 0))
  v = [i.xyz for i in obj.vertices]
  plot_points_3D(v)
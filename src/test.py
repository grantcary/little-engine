from primatives import Point, Vertex, Face, Object
from matplotlib import pyplot as plt
import random
import copy

def complex_edge_mesh():
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

def plot_points_3D(points: list):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in points:
    ax.scatter(i[0], i[1], i[2], marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

if __name__ == "__main__":
  obj = square_pyramid_generator(4, 4)
  obj.moveto(Point(0, 0, 0))
  v = [i.xyz for i in obj.vertices]
  plot_points_3D(v)
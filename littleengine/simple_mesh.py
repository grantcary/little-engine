from matplotlib import pyplot as plt
from read_file import OBJ
import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

class Object():
  def __init__(self, name = 'none', vertices = np.array([]), faces = np.array([])):
      self.name = name
      self.vertices = vertices
      self.f_to_v = {i: list(verts) for i, verts in enumerate(faces)} # maps what vertices a given face uses

      d = {}
      for f, v in self.f_to_v.items():
        for i in v:
          if i not in d:
            d[i] = []
          d[i].append(f)
      self.v_to_f = dict(sorted(d.items())) # maps what faces use a given vertex

  def translate(self, x, y, z):
    self.vertices += np.array([x, y, z])

  def vertex_faces(self, vertex):
    return np.array(self.v_to_f[vertex])
  
  def face_vertices(self, face):
    return np.array([self.vertices[v] for v in self.f_to_v[face]])


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
  vertices, faces = OBJ(OBJ_FILE).read()
  o = Object('test', np.array(vertices), np.array(faces))
  # print(o.vertices)
  # print(o.faces) 

  plot_points_3D(o.vertices)
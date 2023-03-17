from matplotlib import pyplot as plt
from read_file import OBJ
import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

class Object():
  def __init__(self, name = 'none', vertices = np.array([]), faces = None, origin = None):
      self.name = name
      self.vertices = vertices # list of raw vertices
      if faces:
        self.f_to_v = {i: list(verts) for i, verts in enumerate(faces)} # maps what vertices a given face uses
        d = {}
        for f, v in self.f_to_v.items():
          for i in v:
            if i not in d:
              d[i] = []
            d[i].append(f)
        self.v_to_f = dict(sorted(d.items())) # maps what faces use a given vertex
      self.origin = self.set_origin() if origin is None else origin # center of object based on max and min vertices on each axis

  # takes max and min of each axis and averages them to find the center
  def set_origin(self):
    x = (max(self.vertices, key=lambda p: p[0])[0] + min(self.vertices, key=lambda p: p[0])[0]) / 2
    y = (max(self.vertices, key=lambda p: p[1])[1] + min(self.vertices, key=lambda p: p[1])[1]) / 2
    z = (max(self.vertices, key=lambda p: p[2])[2] + min(self.vertices, key=lambda p: p[2])[2]) / 2
    return np.array([x, y, z])

  def translate(self, x, y, z):
    self.vertices += np.array([x, y, z])

  # move to a specific point
  def moveto(self, x, y, z):
    target = np.array([x, y, z])
    dx, dy, dz = target - self.origin
    for v in self.vertices:
      v += np.array([dx, dy, dz])
    self.origin = target

  def vertex_faces(self, vertex):
    return np.array(self.v_to_f[vertex])
  
  def face_vertices(self, face):
    return np.array([self.vertices[v] for v in self.f_to_v[face]])
  
  def __repr__(self):
    return f"Object(name: {self.name}, vertices: {len(self.vertices)}, faces: {len(self.f_to_v)})"


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
  print(repr(o))

  # plot_points_3D(o.vertices)
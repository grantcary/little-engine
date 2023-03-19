from matplotlib import pyplot as plt
from read_file import OBJ
import numpy as np
from time import time

OBJ_FILE = '../test_objects/suzie.obj'

class Object():
  def __init__(self, name = None, vertices = None, faces = None, origin = None):
      self.name = name
      self.vertices = vertices # list of raw vertices
      # TODO: 'if faces' not working as expected, requires 'if faces is not None'. FIX
      self.face_map = {i: list(verts) for i, verts in enumerate(faces)} if faces is not None else None # maps what vertices make up a given face
      if faces is not None:
        vertex_keys = {i: [] for i in range(faces.flatten().max() + 1)} # generates a dictionary with keys for each vertex index
        for i in range(len(faces)):
          for j in faces[i]:
            vertex_keys[j].append(i)
      self.vertex_map = dict(sorted(vertex_keys.items())) if faces is not None else None # maps adjacent faces to each vertex
      self.origin = self.set_origin() if origin is None else origin # center of object based on max and min vertices on each axis

  # takes max and min of each axis and averages them to find the center
  def set_origin(self):
    x = (max(self.vertices, key=lambda p: p[0])[0] + min(self.vertices, key=lambda p: p[0])[0]) / 2
    y = (max(self.vertices, key=lambda p: p[1])[1] + min(self.vertices, key=lambda p: p[1])[1]) / 2
    z = (max(self.vertices, key=lambda p: p[2])[2] + min(self.vertices, key=lambda p: p[2])[2]) / 2
    return np.array([x, y, z])

  # TODO: start linear algebra
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
    return np.array(self.vertex_map[vertex]) if self.vertex_map is not None else None
  
  def face_vertices(self, face):
    return np.array([self.vertices[v] for v in self.face_map[face]]) if self.face_map is not None else None
  
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
  obj = OBJ(OBJ_FILE)
  vertices, faces = obj.read()
  o = Object(name = obj.name, vertices = np.array(vertices), faces = np.array(faces))
  # print('face to vertex', o.face_map)
  # print('vertex to face', o.vertex_map)

  # plot_points_3D(o.vertices)
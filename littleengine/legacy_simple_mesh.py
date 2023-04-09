from matplotlib import pyplot as plt
import numpy as np
from typing import List
from time import time

class Object():
  def __init__(self, name: str = None, vertices: List = None, faces: List = None, origin: np.array = None):
      self.name = name
      self.vertices = vertices if vertices is not None else []
      self.face_map = self.compose_face_map(faces) # maps what vertices make up a given face
      self.vertex_map = self.compose_vertex_map(faces) # maps adjacent faces to each vertex
      self.origin = self.set_origin() if origin is None else origin # center of object based on max and min vertices on each axis

  def compose_face_map(self, faces: List) -> dict:
      if faces is None:
          return None
      return {i: list(verts) for i, verts in enumerate(faces)}

  def compose_vertex_map(self, faces: List) -> dict:
      if faces is None:
          return None
      return {vertex: [i for i, face in enumerate(faces) if vertex in face] for vertex in range(np.max(faces) + 1)}

  # takes max and min of each axis and averages them to find the center
  def set_origin(self):
    vertices_array = np.array(self.vertices)
    min_coords = vertices_array.min(axis=0)
    max_coords = vertices_array.max(axis=0)
    return (min_coords + max_coords) / 2

  # TODO: start linear algebra
  def translate(self, x, y, z):
    self.vertices += np.array([x, y, z])

  # move to a specific point
  def moveto(self, x, y, z):
    target = np.array([x, y, z])
    displacement = target - self.origin
    self.vertices = (np.array(self.vertices) + displacement).tolist()
    self.origin = target

  def vertex_faces(self, vertex):
    return np.array(self.vertex_map[vertex]) if self.vertex_map is not None else None
  
  def face_vertices(self, face):
    return np.array([self.vertices[v] for v in self.face_map[face]]) if self.face_map is not None else None
  
  def __repr__(self):
    return f"Object(name: {self.name}, vertices: {len(self.vertices)}, faces: {len(self.face_map)})"
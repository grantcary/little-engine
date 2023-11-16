from matplotlib import pyplot as plt
import numpy as np
from typing import List

class SceenParams():
  def __init__(self, width: int = 100, height: int = 100, max_ray_depth: int = 3, use_bvh: bool = False, background_color: List[int] = [6, 20, 77]):
    self.w = width
    self.h = height
    self.bgc = background_color
    self.depth = max_ray_depth
    self.use_bvh = use_bvh

def rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))

def plot_points_3D(obj, labels : bool = False):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i, p in enumerate(obj):
    ax.scatter(p[0], p[1], p[2], marker='o')
    if labels:
      ax.text(p[0], p[1], p[2], f'{i}, ({p[0]}, {p[1]}, {p[2]})')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

def plot_surface_3D(coords: list):
  grid_shape = (10, 10)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  X = coords[:, 0].reshape(grid_shape)
  Y = coords[:, 1].reshape(grid_shape)
  Z = coords[:, 2].reshape(grid_shape)

  ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
  ax.set_aspect('equal')

  plt.show()

def plot_vectors_3D(vectors):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for vector in vectors:
    ax.quiver(0, 0, 0, *vector, arrow_length_ratio=0.1)

  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([-1, 1])

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()
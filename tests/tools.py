from matplotlib import pyplot as plt
import numpy as np

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

def surface_function(x, y):
    # Example surface: z = x^2 + y^2
    return x**2 + y**2

def plot_surface_3D(coords: list):
  grid_shape = (10, 10)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  X = coords[:, 0].reshape(grid_shape)
  Y = coords[:, 1].reshape(grid_shape)
  Z = coords[:, 2].reshape(grid_shape)


  # ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
  ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
  ax.set_aspect('equal')

  plt.show()

def plot_vectors_3D(obj):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for vector in obj.normals:
    ax.quiver(0, 0, 0, *vector, arrow_length_ratio=0.1)

  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([-1, 1])

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()
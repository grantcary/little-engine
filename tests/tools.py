from matplotlib import pyplot as plt
import numpy as np

def plot_points_3D(points: list):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in points:
    ax.scatter(i[0], i[1], i[2], marker='o')

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
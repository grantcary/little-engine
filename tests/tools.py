from matplotlib import pyplot as plt

def plot_points_3D(points: list):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in points:
    ax.scatter(i[0], i[1], i[2], marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()
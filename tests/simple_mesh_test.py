import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.simple_mesh import Object as SimpleObject
from littleengine.read_file import OBJ
from tools import plot_points_3D
import numpy as np

OBJ_FILE = '../test_objects/default_cube.obj'

if __name__ == "__main__":
  obj = OBJ()
  vertices, faces = obj.read(OBJ_FILE)
  o = SimpleObject(name = obj.name, vertices = np.array(vertices), faces = np.array(faces))
  print(o)

  # print('face to vertex', o.face_map)
  # print('vertex to face', o.vertex_map)

  # plot_points_3D(o.vertices)
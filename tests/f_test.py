import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.read_file import OBJ
from littleengine.legacy_mesh import Group
from tools import plot_points_3D

OBJ_FILE_1 = '../test_objects/icosphere.obj'
OBJ_FILE_2 = '../test_objects/default_cube.obj'
OBJ_FILE_3 = '../test_objects/suzie.obj'

def group_test():
  ico_file = OBJ(OBJ_FILE_1)
  cube_file = OBJ(OBJ_FILE_2)
  ico = ico_file.get_object()
  cube = cube_file.get_object()
  ico.translate(-1, -1, -1)
  cube.translate(1, 1, 1)
  g = Group('ico-cube', ico, cube)
  v = [i.xyz for o in g.objects for i in o.vertices]
  plot_points_3D(v)

def large_object_test():
  suzie_file = OBJ(OBJ_FILE_3)
  suzie = suzie_file.get_object()
  v = [i.xyz for i in suzie.vertices]
  plot_points_3D(v)

if __name__ == "__main__":
  group_test()
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.read_file import OBJ
from littleengine.mesh import Group
from tools import plot_points_3D

OBJ_FILE_1 = '../test_objects/icosphere.obj'
OBJ_FILE_2 = '../test_objects/default_cube.obj'

ico_file = OBJ(OBJ_FILE_1)
cube_file = OBJ(OBJ_FILE_2)
ico = ico_file.get_object()
cube = cube_file.get_object()
cube.translate(0, 0, 3)
g = Group('ico-cube', ico, cube)
v = [i.xyz for o in g.objects for i in o.vertices]
plot_points_3D(v)
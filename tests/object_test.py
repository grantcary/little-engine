import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.object import Object
import tools

OBJ_FILE = '../test_objects/icosphere.obj'

o = Object('Cube', OBJ_FILE)
tools.plot_points_3D(o)
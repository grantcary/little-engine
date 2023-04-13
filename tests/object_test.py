import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine.object import Object

OBJ_FILE = '../test_objects/default_cube.obj'

o = Object('Cube', OBJ_FILE)
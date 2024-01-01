import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from littleengine import Object

CUBE = 'cube.obj'
c = Object('Cube', CUBE)
for n in c.normals:
    print(n)
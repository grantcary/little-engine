import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from littleengine.object import Object
from littleengine.read_file import OBJ
import tools

OBJ_FILE = '../test_objects/suzie.obj'

o = Object('Cube', OBJ_FILE)
v_size = len(o.vertices)
vertex_samples = np.random.choice(v_size, v_size // 2, replace=False)
# print(vertex_samples)
tools.plot_points_3D(o.vertices[vertex_samples])
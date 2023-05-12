import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

np.set_printoptions(threshold=sys.maxsize)

import littleengine.object as object
from littleengine.meshlet import meshlet_gen, generate_triangle_adjacency
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

suzie = object.Object('Monkey', SUZIE)

st = time.time()
meshlet_gen(suzie)
print(time.time() - st)
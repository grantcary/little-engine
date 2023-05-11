import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import littleengine.object as object
from littleengine.meshlet import generate_triangle_adjacency, compute_triangle_cones
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'

suzie = object.Object('Monkey', SUZIE)

t, n, a = compute_triangle_cones(suzie.vertices, suzie.faces)

# print(kdtreeBuild(suzie.vertices))
# BUILD KDTREE
# distance, index = cKDTree(suzie.vertices).query(np.array([0.5, 0.5, 0.5]))
# print(distance, index, suzie.vertices[index])

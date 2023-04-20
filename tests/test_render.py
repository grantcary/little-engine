import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import littleengine.mesh as mesh
import littleengine.object as object
import littleengine.render as render
import littleengine.camera as camera
import tools


OBJ_FILE = '../test_objects/default_cube.obj'

o = object.Object('Cube', OBJ_FILE)

cam = camera.Camera(90, aspect_ratio=1)
cam.translate(0, 0, 5)
v = render.camera_ray_test(cam)
t = render.trace(o, cam.position, v)
# print(t, t.shape)
# print(np.unique(t, axis=0))

# m = mesh.Mesh(None, None, v)
# tools.plot_vectors_3D(m)
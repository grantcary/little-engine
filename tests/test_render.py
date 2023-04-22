import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import littleengine.mesh as mesh
import littleengine.object as object
import littleengine.camera as camera
import littleengine.render as render
import tools

OBJ_FILE = '../test_objects/suzie.obj'

obj = object.Object('Cube', OBJ_FILE)

cam = camera.Camera(90, aspect_ratio=1)
cam.translate(0, -2.5, 0)
cam.rotation = np.array([0, 0, 0]) 

render.render(25, 25, cam, obj)

# rays = render.camera_ray_test(25, 25, cam)
# m = mesh.Mesh(None, None, rays)
# tools.plot_vectors_3D(m)
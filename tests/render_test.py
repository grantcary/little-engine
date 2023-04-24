import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import littleengine.mesh as mesh
import littleengine.object as object
import littleengine.camera as camera
import littleengine.render as render
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'

scene = []
suzie = object.Object('Monkey', SUZIE)
cube = object.Object('Light', CUBE)
cube.mesh.translate(0, 5, 0)
scene.append(suzie)
scene.append(cube)

cam = camera.Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 10])
cam.rotation = np.array([0, 180, 0]) 

render.render(200, 200, cam, scene)

# rays = render.camera_ray_test(25, 25, cam)
# m = mesh.Mesh(None, None, rays)
# tools.plot_vectors_3D(m)
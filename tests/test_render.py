import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import littleengine.mesh as mesh
import littleengine.render as render
import littleengine.camera as camera
import tools

cam = camera.Camera(90, aspect_ratio=1)
# cam.translate(0, 0, 5)
v = render.camera_ray_test(cam)
m = mesh.Mesh(None, None, v)
tools.plot_vectors_3D(m)

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from littleengine import Camera, Object, Light, SceenParams, render_experimental, Skybox
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'
PLANE = '../test_objects/plane.obj'

USE_BVH = False

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], rotate=[90, 0, 0], color=[255, 0, 0], reflectivity=0.01, bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], reflectivity=1, ior=1.3, bvh=USE_BVH)
icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, reflectivity=0.2, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, bvh=USE_BVH)
plane = Object('Plane', PLANE, position=[0, 0, -0.3], scale=7, rotate=[100, 0, 0], color=[0, 0, 0], reflectivity=1, bvh=USE_BVH)
objects = [suzie, cube, plane]

spherical_1 = Light('Spherical 1', position=[-3, -3, 1], intensity=1)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))

params = SceenParams(400, 400, [6, 20, 77], 10, USE_BVH)
skybox = Skybox('../littleengine/textures/puresky.jpg')
images = []
render_times = []
count = 1
start_pos = 6
end_pos = -6
total_frames = 60
st = time.time()
for i in np.linspace(start_pos, end_pos, total_frames):
    print(f'IMAGE {count + 1} / {total_frames}')
    swing_y = 0.15 * i**2 - 5
    pan_z = i * 10 + 180
    cam = Camera(position=[i, swing_y, 0], rotation=[90, 0, pan_z], fov=90, aspect_ratio=1)

    image, render_time = render_experimental(params, cam, skybox, objects, lights)
    print(f'Frame {count} Render Time: {render_time:.4f}s')
    images.append(image)
    render_times.append(render_time)
    count += 1
print(f'Total Animation Render Time: {time.time() - st}')

images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

plt.figure()
plt.plot(range(0, count - 1), render_times)
plt.xlabel('Frame')
plt.ylabel('Render Time')
plt.title('Animation Render Report')
plt.show()
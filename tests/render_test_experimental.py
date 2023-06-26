import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from littleengine import Camera, Object, Light, SceenParams, render_experimental
import tools

SUZIE = '../test_objects/suzie.obj'
CUBE = '../test_objects/default_cube.obj'
ICOSPHERE = '../test_objects/icosphere.obj'
TEAPOT = '../test_objects/teapot.obj'

USE_BVH = True

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], color=[255, 0, 0], reflectivity=0.3, bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, bvh=USE_BVH)
icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, reflectivity=0.2, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, bvh=USE_BVH)
objects = [suzie, cube]

spherical_1 = Light('Spherical 1', position=[0, 3, 3], intensity=1.0)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)


params = SceenParams(200, 200, [6, 20, 77], 3, USE_BVH)
images = []
render_times = []
count = 0
start_pos = 6
end_pos = -6
total_frames = 80
st = time.time()
for i in np.linspace(start_pos, end_pos, total_frames):
    print(f'IMAGE {count + 1} / {total_frames}')
    cam = Camera(position=[i, 0, 3], rotation=[0, 180, 0], fov=90, aspect_ratio=1)

    # print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
    image, render_time = render_experimental(cam, params, objects, lights)
    print(f'Frame {count + 1} Render Time: {render_time:.4f}s')
    images.append(image)
    render_times.append(render_time)
    count += 1
print(f'Total Animation Render Time: {time.time() - st}')

images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

plt.figure()
plt.plot(range(0, count), render_times)
plt.xlabel('Frame')
plt.ylabel('Render Time')
plt.title('Animation Render Report')
plt.show()
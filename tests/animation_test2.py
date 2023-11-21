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

teapot = Object('Teapot', TEAPOT, position=[12, -5.5, 0], rotate=[90, 0, 90], scale=0.35, color=[134, 229, 231], reflectivity=0.25)
icosphere = Object('Icosphere', ICOSPHERE, position=[5, -2, 1], rotate=[90, 0, 90], color=[134, 229, 231], reflectivity=0.25)
cube1 = Object('Cube 1', CUBE, position=[-0.75, -2, 1], rotate=[0, 0, 45], scale=1.6, color=[0, 0, 0], reflectivity=1)
cube2 = Object('Cube 2', CUBE, position=[1.5, -4, 1], color=[231, 231, 231])
cube3 = Object('Cube 3', CUBE, position=[2, 1, 1], color=[231, 231, 231])
cube4 = Object('Cube 4', CUBE, position=[-2, -1, 1], color=[231, 231, 231])
cube5 = Object('Cube 5', CUBE, position=[-2, 1, 1], color=[231, 231, 231])
plane = Object('Plane', PLANE, position=[0, 0, 0], scale=7, rotate=[90, 0, 0], color=[231, 231, 231])
objects = [icosphere, cube1, cube2, cube3, cube4, cube5, plane]

spherical_1 = Light('Spherical 1', position=[0, 0, 4], intensity=1)
lights = [spherical_1]

# dot = tools.add_nodes_edges(suzie.bvh)
# dot.render('binary_tree.gv', view=True)

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))

params = SceenParams(400, 300, 3)
skybox = Skybox('../littleengine/textures/puresky.jpg')
images = []
render_times = []
count = 1
start = 0
end = 36
frames = 120
print(f'Render start time: {time.strftime("%H:%M", time.localtime())}')
st = time.time()

def parallax(keyframes, degrees, degrees_offset=90, x=0, y=0, rad=5):
    # rot = tools.linearscale(keyframes, 0, degrees / len(keyframes))
    rot = np.linspace(0, degrees, len(keyframes))
    pos = np.array(tools.circumference(rot + degrees_offset, x, y, rad))
    return pos, rot

def ease(keyframes, strength=10):
    normalized = np.linspace(0, 1, len(keyframes))
    sigmoid = 1 / (1 + np.exp(-strength * (normalized - 0.5)))
    scaled = sigmoid * (keyframes[0] - keyframes[-1]) + keyframes[-1]
    return np.flip(scaled)

def ease_in(keyframes):
    return keyframes ** 2

def ease_out(keyframes):
    return 1 - (1 - keyframes) ** 2

keyframes = np.linspace(start, end, frames)
move = ease(np.linspace(5.5, 0, frames))
dolly = ease(np.linspace(7, 5.5, frames))
pos, rot = parallax(ease(keyframes), 90, rad=5.5)
zoom = np.concatenate((ease(np.linspace(50, 85, 90)), np.full((30,), 85)))
rot3 = np.concatenate((ease(np.linspace(90, 125, 90)), np.full((30, ), 125)))
crane = np.concatenate((ease(np.linspace(1, 5, 90)), np.full((30, ), 5)))

# move 1
for i in range(60):
    print(f'IMAGE {count} / {frames + 60}')
    cam = Camera(position=[0, dolly[i], 1], rotation=[90, 0, 0], fov=50)
    image, render_time = render_experimental(params, cam, skybox, objects, lights)
    images.append(image)
    render_times.append(render_time)
    estimated_time = (sum(render_times) / len(render_times)) * (frames + 60 - count + 1)
    print(f'Frame {count} | Render Time: {render_time:.4f}s | Time Left: {int(estimated_time//60)} minutes')
    count += 1

# move 2
for i in range(frames):
    print(f'IMAGE {count} / {frames + 60}')
    cam = Camera(position=[pos[0, i], pos[1, i], crane[i]], rotation=[rot3[i], 0, rot[i]], fov=zoom[i])
    image, render_time = render_experimental(params, cam, skybox, objects, lights)
    images.append(image)
    render_times.append(render_time)
    estimated_time = (sum(render_times) / len(render_times)) * (frames + 60 - count + 1)
    print(f'Frame {count} | Render Time: {render_time:.4f}s | Time Left: {int(estimated_time//60)} minutes')
    count += 1

print(f'Total Animation Render Time: {int((time.time() - st)//60)} minutes')

images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

plt.figure()
plt.plot(range(0, count - 1), render_times)
plt.xlabel('Frame')
plt.ylabel('Render Time')
plt.title('Animation Render Report')
plt.show()
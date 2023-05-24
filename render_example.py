import numpy as np

from littleengine import *

SUZIE = 'test_objects/suzie.obj'
CUBE = 'test_objects/default_cube.obj'
ICOSPHERE = 'test_objects/icosphere.obj'

objects = []
suzie = Object('Monkey', SUZIE)
suzie.material_type = 'diffuse'
suzie.color = np.array([255, 0, 0])
suzie.translate(0, 0, 0)
objects.append(suzie)

cube = object.Object('Cube', CUBE)
cube.material_type = 'diffuse'
cube.translate(2, 0, 0)
cube.color = np.array([0, 255, 0])
# objects.append(cube)

lights = []
ico = Object('Light', ICOSPHERE)
ico.material_type = 'emissive'
ico.translate(0, 3, 3)
lights.append(ico)

cam = Camera(90, aspect_ratio=1)
cam.position = np.array([0, 0, 5])
cam.rotation = np.array([0, 180, 0])

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
render(100, 100, cam, objects, lights)
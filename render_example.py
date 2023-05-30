from littleengine import *

SUZIE = 'test_objects/suzie.obj'
CUBE = 'test_objects/default_cube.obj'
ICOSPHERE = 'test_objects/icosphere.obj'

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], color=[255, 0, 0], reflectivity=0.3)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0])
objects = [suzie, cube]

spherical_1 = Light('Spherical 1', position=[0, 3, 3], intensity=1.0)
lights = [spherical_1]

cam = Camera(position=[0, 0, 5], rotation=[0, 180, 0], fov=90, aspect_ratio=1)

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
render(100, 100, cam, objects, lights)
from littleengine import Camera, Object, Light, SceenParams, render

SUZIE = 'test_objects/suzie.obj'
CUBE = 'test_objects/default_cube.obj'
ICOSPHERE = 'test_objects/icosphere.obj'

suzie1 = Object('Monkey', SUZIE, position=[2, 0, 0], color=[255, 0, 0], reflectivity=0.3)
suzie2 = Object('Monkey', SUZIE, position=[-2, 0, 0], color=[0, 255, 0])
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)
icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3)
objects = [suzie1, icosphere]

spherical_1 = Light('Spherical 1', position=[0, 3, 3], intensity=1.0)
lights = [spherical_1]

cam = Camera(position=[0, 0, 5], rotation=[0, 180, 0], fov=90, aspect_ratio=1)
params = SceenParams(100, 100, [6, 20, 77], 3)

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
render(cam, params, objects, lights)
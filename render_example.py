from littleengine import Camera, Skybox, Object, Light, SceenParams, render

SUZIE = 'test_objects/suzie.obj'
CUBE = 'test_objects/default_cube.obj'
ICOSPHERE = 'test_objects/icosphere.obj'
TEAPOT = 'test_objects/teapot.obj'
PLANE = 'test_objects/plane.obj'

USE_BVH = False

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], rotate=[90, 0, 0], color=[255, 0, 0], reflectivity=0.01, bvh=USE_BVH)
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], reflectivity=0.01, ior=1.3, bvh=USE_BVH)
icosphere = Object('Icosphere', ICOSPHERE, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, reflectivity=0.2, bvh=USE_BVH)
teapot = Object('Teapot', TEAPOT, position=[-2, 0, 0], color=[0, 255, 0], ior=1.3, bvh=USE_BVH)
plane = Object('Plane', PLANE, position=[0, 0, -0.3], scale=7, rotate=[100, 0, 0], color=[0, 0, 0], reflectivity=1, bvh=USE_BVH)
objects = [suzie, cube, plane]

spherical_1 = Light('Spherical 1', position=[-3, -3, 1], intensity=1)
lights = [spherical_1]

params = SceenParams(1920, 1080, [6, 20, 77], 10, USE_BVH)
cam = Camera(position=[0, -5, 0], rotation=[90, 0, 180], fov=90, aspect_ratio=1)
skybox = Skybox('littleengine/textures/puresky.jpg')

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
image, render_time = render(params, cam, skybox, objects, lights)
print(f'Total Render Time: {render_time:.3f}s')
image.show()
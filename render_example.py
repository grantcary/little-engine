from littleengine import Camera, Skybox, Object, Light, SceenParams, render

SUZIE = 'test_objects/suzie.obj'
CUBE = 'test_objects/default_cube.obj'
PLANE = 'test_objects/plane.obj'

suzie = Object('Monkey', SUZIE, position=[2, 0, 0], rotate=[90, 0, 0], color=[255, 0, 0])
cube = Object('Cube', CUBE, position=[-2, 0, 0], color=[0, 255, 0], reflectivity=0.8)
plane = Object('Plane', PLANE, position=[0, 0, -0.3], scale=7, rotate=[100, 0, 0], color=[0, 0, 0], reflectivity=1)
objects = [suzie, cube, plane]

spherical_1 = Light('Spherical 1', position=[-3, -3, 3], intensity=1)
lights = [spherical_1]

params = SceenParams(400, 300, 5)
cam = Camera(position=[0, -5, 0], rotation=[90, 0, 180], fov=90)
skybox = Skybox('littleengine/textures/puresky.jpg')

print('Total Triangles in Scene:', sum([len(o.faces) for o in objects]))
image, render_time = render(params, cam, skybox, objects, lights)
print(f'Total Render Time: {render_time:.3f}s')
image.show()
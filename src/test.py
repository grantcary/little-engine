from primatives import Point, Vertex, Object
import random

def create_object():
  vertices = []
  for i in range(10):
    p = Point(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11))
    vertices.append(Vertex([p, p, p]))
    print(f"{i}: {p.x}, {p.y}, {p.z}")
  return Object(vertices)
obj = create_object()
p = obj.origin
print(f"Origin: {p.x}, {p.y}, {p.z}")

print("\nMoving object to random point\n")

obj.moveto(Point(random.randrange(-10, 11), random.randrange(-10, 11), random.randrange(-10, 11)))
p = obj.origin
for i in obj.vertices:
  print(f"{i.points[0].x}, {i.points[0].y}, {i.points[0].z}")
print(f"Origin: {p.x}, {p.y}, {p.z}")

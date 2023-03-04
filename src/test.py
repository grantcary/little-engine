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
p = obj.find_origin()
print(f"Origin: {p.x}, {p.y}, {p.z}")

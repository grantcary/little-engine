from littleengine.mesh import Vertex, Face, Object

class OBJ():
  def __init__(self, filename):
    self.name = None
    with open(filename, "r") as f:
      self.lines = f.read().splitlines()
    self.vertices, self.faces = self.read()


  def read(self) -> list:
    vertices, faces = [], []
    for line in self.lines:
      if line:
        prefix, value = line.split(" ", 1)
        if prefix == "o":
          self.name = value
        elif prefix == "v":
          pos = list(map(float, value.split(" ")))
          vertices.append(pos if len(pos) == 3 else pos[:3])
        elif prefix == "f":
          faces.append([int(face.split("/")[0]) - 1 for face in value.split(" ")])
    return vertices, faces
  
  def get_object(self) -> Object:
    vertices, faces = [], []
    for v in self.vertices:
      vertices.append(Vertex(*v))
    for f in self.faces:
      faces.append(Face(*[vertices[i] for i in f]))
    for i in range(len(vertices)):
      vertices[i].faces = [f for f in faces if vertices[i] in f.vertices]
    return Object(self.name, vertices, faces)
      
  def __len__(self):
    return len(self.lines)
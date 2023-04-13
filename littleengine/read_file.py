import numpy as np

class OBJ():
  def __init__(self, name: str = None):
    self.name = "unnamed"

  def read(self, filepath: str) -> list[list, list]:
    with open(filepath, "r") as f:
      lines = f.read().splitlines()
    vertices, faces = [], []
    for line in lines:
      if line:
        prefix, value = line.split(" ", 1)
        if prefix == "o":
          self.name = value
        elif prefix == "v":
          pos = list(map(float, value.split(" ")))
          vertices.append(pos if len(pos) == 3 else pos[:3])
        elif prefix == "f":
          faces.append([int(face.split("/")[0]) - 1 for face in value.split(" ")])

    max_length = max(len(sublist) for sublist in faces)
    padded_list = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in faces]

    # TODO: change dtype back to object once ear clipping algorithm is implemented 
    return np.array(vertices), np.array(padded_list, dtype=np.int64)
  
  def write(self, path: str, mesh) -> None:
    obj_data = []
    obj_data.append('# OBJ file generated by little-engine')
    obj_data.append('# https://github.com/grantcary/little-engine')
    obj_data.append(f"o {self.name}")

    for v in mesh.vertices:
      obj_data.append(f"v {' '.join([f'{num:.6f}' for num in v])}")
    for v in mesh.faces:
      obj_data.append(f"f {' '.join([f'{num + 1}' for num in mesh.faces[v]])}")
    obj_data = ('\n').join(obj_data)

    with open(path, 'w') as f:
      f.write(obj_data)
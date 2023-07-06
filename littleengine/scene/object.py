from littleengine.scene import mesh
from littleengine.experimental.meshlet import meshlet_gen
from littleengine.experimental.bvh import bounding_volume_hierarchy
import numpy as np
# from scipy.spatial.transform import Rotation as R

class Object:
    def __init__(self, name=None, path=None, position=[0.0, 0.0, 0.0], rotate=[0.0, 0.0, 0.0], 
                 scale=1, color=[127, 127, 127], luminance=0.0, reflectivity=0.0, ior=0.0,
                 bvh=False):
        self.name = name
        self.position = np.array(position, dtype=np.float32)
        self.mesh = mesh.Mesh(*self.read_file(path))
        self.mesh.triangulate()
        self.mesh.rotate(rotate)
        self.mesh.translate(*position)
        self.scale(scale)

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.normals = self.mesh.normals

        self.color = np.array(color, dtype=np.uint8)
        self.luminance = luminance
        self.reflectivity = reflectivity
        self.ior = ior

        self.bvh = self.meshlet_bvh() if bvh else None

    def read_file(self, path):
        with open(path, "r") as f:
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

        return np.array(vertices, dtype=np.float64), np.array(padded_list, dtype=np.int64)

    def write_file(self, output_path: str):
        obj_data = []
        obj_data.append('# OBJ file generated by little-engine')
        obj_data.append('# https://github.com/grantcary/little-engine')
        obj_data.append(f"o {self.name}")

        for v in self.vertices:
            obj_data.append(f"v {' '.join([f'{num:.6f}' for num in v])}")
        for f in self.faces:
            obj_data.append(f"f {' '.join([f'{num + 1}' for num in f])}")
        obj_data = ('\n').join(obj_data)

        with open(output_path, 'w') as f:
            f.write(obj_data)

    def translate(self, x, y, z):
        self.position += np.array([x, y, z])
        self.mesh.translate(x, y, z)

    def scale(self, s):
        self.mesh.vertices *= s

    def rotate(self, rotation):
        self.mesh.translate(*-self.position)
        self.mesh.rotate(rotation)
        self.mesh.translate(*self.position)
        self.vertices = self.mesh.vertices
        self.normals = self.mesh.normals

    def meshlet_bvh(self, sort_meshlets=True, sort_axis=2):
        meshlets = meshlet_gen(self)
        if sort_meshlets:
            meshlets.sort(key=lambda meshlet: meshlet.centroid[sort_axis])
        return bounding_volume_hierarchy(self, meshlets)
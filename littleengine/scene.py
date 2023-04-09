from littleengine.legacy_mesh import Object, Group

class Scene():
  def __init__(self, name: str, objects: list[Object] = [], groups: list[Group] = []):
    self.name = name
    self.objects = objects
    self.groups = groups
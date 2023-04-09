import time
from littleengine.legacy_mesh import Object, Point

class Animation():
  def __init__(self, frames: int, fps: int, loop: bool = False, keyframes: list[dict[str, Point]] = [], target_object: Object = None):
    self.frames = frames
    self.fps = fps
    self.loop = loop
    self.frame_duration = 1 / fps
    self.keyframes = keyframes # [{timestamp: '00:00.000', position: Point(x, y, z)], {timestamp: '00:00.500', position: Point(x, y, z)}, {timestamp: '00:01.000', position: Point(x, y, z)}]
    self.start_frame = 0
    self.current_frame = 0
    self.last_frame = len(keyframes) - 1
    self.object = target_object


  def set_keyframe(self, target_frame: int, time: str, point: Point):
    if not target_frame:
      target_frame = self.last_frame + 1
    self.keyframes[target_frame] = {'timestamp': time, 'position': point}

  def process_frame(self):
    self.object.moveto(self.keyframes[self.current_frame]['position'])

  def play_frames(self):
    while self.current_frame <= self.last_frame:
      self.process_frame()
      time.sleep(self.frame_duration)
      self.current_frame += 1
    self.current_frame = 0

# TODO: make transformation wrapper functions for Vertex/Object/Group
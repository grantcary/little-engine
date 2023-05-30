import numpy as np

class Light():
    def __init__(self, name=None, position=[0.0, 0.0, 0.0], intensity=1.0, color=[0, 0, 0]):
        self.name = name
        self.position = np.array(position, dtype=np.float32)
        self.intensity = intensity
        self.color = np.array(color, dtype=np.uint8)
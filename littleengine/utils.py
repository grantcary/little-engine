import numpy as np

def rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))
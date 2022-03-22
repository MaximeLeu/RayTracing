import numpy as np


class Vector:
    def __init__(self, orig, dest=None, direction=None):
        self.orig = np.asarray(orig, dtype=float).reshape(-1)

        if dest is not None:
            self.dest = np.asarray(dest, dtype=float).reshape(-1)
            self.direction = self.dest - self.orig
        elif direction is not None:
            self.direction = np.asarray(direction, dtype=float).reshape(-1)
            self.dest = self.orig + self.direction
        else:
            raise ValueError("`dest` and `direction` cannot be None simultaneously.")

        self.len = np.linalg.norm(self.direction)
        self.unit = self.direction / self.len

    def at(self, t):
        return self.orig + t * self.direction

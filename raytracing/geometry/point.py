import numpy as np

from ..plotting import Plotable, draw_points


class Point(Plotable):
    def __init__(self, point):
        super().__init__()

        self.point = np.asarray(point, dtype=float).reshape(3)
        self.domain = np.array([self.point, self.point])

    def plot(self, *args, **kwargs):

        draw_points(self.ax, self.point.reshape(1, 3), *args, **kwargs)

        return self.ax

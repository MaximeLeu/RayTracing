import numpy as np

from ..plotting import Plotable, draw_line


class Path(Plotable):
    def __init__(self, points):
        super().__init__()

        self.points = np.asarray(points, dtype=float).reshape(-1, 3)

    def plot(self, *args, **kwargs):
        draw_line(self.ax, self.points, *args, **kwargs)
        return self.ax

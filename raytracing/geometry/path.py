import numpy as np

from ..interaction import Edge
from ..plotting import Plotable, draw_line

COLORS = [
    "k",
    "y",
    "r",
    "m",
]


class Path(Plotable):
    def __init__(self, points, /, start=None, end=None, interact_list=None, res=None):
        super().__init__()

        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.start = start
        self.end = end
        self.interact_list = interact_list
        self.res = res

    def __len__(self):
        return self.points.shape[0]

    @staticmethod
    def from_optimize_result(res, **kwargs):
        return Path(res.x.reshape(-1, 3), res=res, **kwargs)

    def is_valid(self, tol=1e-5):

        if self.res is not None and abs(self.res.fun) > tol:
            return False

        if self.interact_list is not None:
            for point, geometry in zip(self.points[1:-1, :], self.interact_list):
                if not geometry.contains(point):
                    return False

        return True

    def plot(self, *args, **kwargs):
        color = COLORS[len(self) - 2]
        linestyle = "solid"
        if self.interact_list is not None:
            if any(isinstance(interaction, Edge) for interaction in self.interact_list):
                linestyle = "dashed"
        draw_line(
            self.ax, self.points, *args, color=color, linestyle=linestyle, **kwargs
        )
        return self.ax

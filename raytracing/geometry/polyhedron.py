from base import Geometry
from raytracing.plotting import Plotable

import numpy as np


class Polyhedron(Geometry, Plotable):
    def __init__(self, polygons, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Plotable, self).__init__(**kwargs)

        self.polygons = polygons

    def plot(self, *args, **kwargs):
        for polygon in self.polygons:
            polygon.on(self.ax).plot(*args, **kwargs)

        return self.ax


if __name__ == "__main__":
    from raytracing.plotting import new_3d_axes
    from raytracing.geometry import Polygon

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )

    p1 = Polygon([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5]])
    p2 = Polygon([[0, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]])
    p3 = Polygon([[1, 0, 0], [1, 1, 0], [0.5, 0.5, 0.5]])
    p4 = Polygon([[1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5]])
    p5 = Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    ph = Polyhedron([p1, p2, p3, p4, p5])
    ph.plot()
    ph.show()

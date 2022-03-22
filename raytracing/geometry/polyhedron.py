import numpy as np

from ..plotting import Plotable
from .base import Geometry
from .polygon import Polygon


class Polyhedron(Geometry, Plotable):
    def __init__(self, polygons, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Plotable, self).__init__(**kwargs)

        self.polygons = polygons

    @staticmethod
    def from_2d_polygon(polygon, height=1, keep_ground=True):
        x, y = polygon.exterior.coords.xy
        x = x[:-1]
        y = y[:-1]

        z0 = np.zeros_like(x, dtype=float)
        zh = np.full_like(z0, height)

        bottom_points = np.column_stack([x, y, z0])
        top_points = np.column_stack([x, y, zh])

        top = Polygon(top_points)
        bottom = Polygon(bottom_points)

        n = top_points.shape[0]

        if keep_ground:
            polygons = [top, bottom]
        else:
            polygons = [top]

        bottom_points = bottom_points[
            ::-1, :
        ]  # Bottom points are now oriented cw to match top points

        # For each face other than top and bottom
        for i in range(n):
            A = top_points[i - 1, :]
            B = top_points[i, :]
            C = bottom_points[i - 1, :]
            D = bottom_points[i, :]

            face_points = np.row_stack([A, C, D, B])

            polygon = Polygon(face_points)
            polygons.append(polygon)

        return Polyhedron(polygons)

    def plot(self, *args, **kwargs):
        for polygon in self.polygons:
            polygon.on(self.ax).plot(*args, **kwargs)

        return self.ax

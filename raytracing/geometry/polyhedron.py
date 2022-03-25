from itertools import combinations, product

import numpy as np
from shapely.geometry import Polygon as shPolygon

from ..interaction import LinearEdge
from ..plotting import Plotable
from .base import Geometry, bounding_box
from .polygon import Polygon


class Polyhedron(Geometry, Plotable):
    def __init__(self, polygons, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Plotable, self).__init__(**kwargs)

        self.polygons = polygons
        self.domain = bounding_box([polygon.domain for polygon in polygons])
        self.surfaces = self.polygons
        self.edges = []

        for s1, s2 in combinations(self.polygons, 2):
            equal_edges = (
                e1.join(e2) for e1, e2 in product(s1.edges, s2.edges) if e1 == e2
            )
            edge = next(equal_edges, None)

            if edge:
                self.edges.append(edge)

    @staticmethod
    def from_2d_polygon(polygon, height=1, keep_ground=True):
        if isinstance(polygon, shPolygon):
            x, y = polygon.exterior.coords.xy
            x = x[:-1]
            y = y[:-1]
        elif isinstance(polygon, np.ndarray) or isinstance(polygon, list):
            x, y = np.asarray(polygon, dtype=float).reshape(-1, 2).T

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
            ::1, :
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

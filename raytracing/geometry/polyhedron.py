from itertools import combinations, product

import numpy as np
from shapely.geometry import Polygon as shPolygon

from ..interaction import LinearEdge
from ..plotting import Plotable
from .base import Geometry, bounding_box
from .polygon import Polygon, is_ccw


class Polyhedron(Geometry, Plotable):
    def __init__(self, polygons, drop_concave_edges=True, min_angle=10, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Plotable, self).__init__(**kwargs)

        self.polygons = polygons
        self.domain = bounding_box([polygon.domain for polygon in polygons])
        self.surfaces = self.polygons
        self.edges = []

        max_cos = np.cos(np.deg2rad(min_angle))

        for s1, s2 in combinations(self.polygons, 2):
            equal_edges = (
                e1.join(e2) for e1, e2 in product(s1.edges, s2.edges) if e1 == e2
            )
            edge = next(equal_edges, None)

            if edge:
                # if not (drop_concave_edges and abs(np.dot(s1.normal(), s2.normal())) < max_cos):
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

        if not is_ccw(top_points):
            top_points = top_points[::-1, :]
        else:
            bottom_points = bottom_points[::-1, :]

        top = Polygon(top_points)
        bottom = Polygon(bottom_points)

        # TODO: reverse normal
        """
        if top.get_normal()[2] < 0:  # z component should be positive
            top.parametric = - top.parametric
        if bottom.get_normal()[2] > 0:  # z component should be negative
            bottom.parametric = - bottom.parametric
        """

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

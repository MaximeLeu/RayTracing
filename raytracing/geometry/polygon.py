import numba as nb
import numpy as np
from shapely.geometry import Point as shPoint
from shapely.geometry import Polygon as shPolygon

from ..interaction import Surface
from ..plotting import Plotable, draw_polygon
from .base import Geometry


def plane_st_to_xyz_function(a, b, c, d):
    if a != 0:

        @nb.njit
        def __impl__(s, t):
            x = -(b * s + c * t + d) / a
            return x, s, t

    elif b != 0:

        @nb.njit
        def __impl__(s, t):
            y = -(a * s + c * t + d) / b
            return s, y, t

    elif c != 0:

        @nb.njit
        def __impl__(s, t):
            z = -(a * s + b * t + d) / c
            return s, t, z

    else:
        raise ValueError("a, b, and c cannot be 0 at the same time")
    return __impl__


def plane_xyz_to_st_function(a, b, c, d):
    if a != 0:

        @nb.njit
        def __impl__(x, y, z):
            return y, z

    elif b != 0:

        @nb.njit
        def __impl__(x, y, z):
            return x, z

    elif c != 0:

        @nb.njit
        def __impl__(x, y, z):
            return x, y

    else:
        raise ValueError("a, b, and c cannot be 0 at the same time")
    return __impl__


class Polygon(Geometry, Surface, Plotable):
    def __init__(self, points, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Surface, self).__init__()
        super(Plotable, self).__init__()

        self.points = np.asarray(points, dtype=float)

        if self.points.ndim != 2:
            raise ValueError("The points must be a  2-dimensional array")

        if self.points.shape[1] != 3:
            raise ValueError("The points must be a N-by-3-dimensional array")

        if self.points.shape[0] < 3:
            raise ValueError("A valid polygon must have at least 3 points")

        p0, p1, p2 = self.points[:3, :]
        u = p1 - p0
        u /= np.linalg.norm(u)
        v = p2 - p1
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        self.__matrix = np.column_stack([u, v, w])
        self.__uv = self.__matrix[:, :2]
        self.__polygon = shPolygon(np.dot(self.points, self.__uv))
        self.__normal = w
        d = -np.dot(self.__normal, p0)
        self.parametric = np.append(w, [d])
        self.st_to_xyz_func = plane_st_to_xyz_function(*self.parametric)

    def is_planar(self) -> bool:
        x0 = self.points
        x1 = np.roll(x0, 1, axis=0)
        x2 = np.roll(x1, 1, axis=0)
        v1 = x1 - x0
        v2 = x2 - x1
        normals = np.cross(v1, v2)

        return np.linalg.norm(np.diff(normals, axis=0)) < 1e-6

    def cartesian(self, point):
        d = self.parametric[-1]
        return np.dot(self.__normal, point) + d

    def normal(self, point):
        return self.__normal

    def contains(self, point):
        point = shPoint(np.dot(point[:2], self.uv))
        return self.__polygon.contains(point)

    def plot(self, *args, facecolor=(0, 0, 0, 0), edgecolor="k", alpha=0.1, **kwargs):
        draw_polygon(
            self.ax,
            self.points,
            *args,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            **kwargs
        )
        return self.ax


class Square(Polygon):
    @classmethod
    def from_corners(cls, lower, upper):
        points = np.empty((4, 3), dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        points[0, :] = lower
        points[2, :] = upper
        points[1, :] = np.array([upper[0], lower[1], lower[2]])
        points[3, :] = np.array([lower[0], upper[1], upper[2]])

        return cls(points)

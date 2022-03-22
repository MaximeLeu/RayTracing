import numba as nb
import numpy as np
from base import Geometry
from shapely.geometry import Point as shPoint
from shapely.geometry import Polygon as shPolygon

from raytracing.interaction import Surface
from raytracing.plotting import Plotable, draw_polygon


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


@nb.njit(cache=True)
def orthogonal_matrix(points, normal):
    A = points[0, :]
    B = points[1, :]

    matrix = np.empty((3, 3))
    matrix[0, :] = B - A
    matrix[1, :] = np.cross(normal, matrix[0, :])
    matrix[2, :] = normal  # Already normalized

    for i in range(2):
        matrix[i, :] /= norm([matrix[i, :]])

    return matrix


class Polygon(Geometry, Surface, Plotable):
    def __init__(self, points, **kwargs):
        super(Geometry, self).__init__(**kwargs)
        super(Plotable, self).__init__()

        self.points = np.asarray(points, dtype=float)

        if self.points.ndim != 2:
            raise ValueError("The points must be a  2-dimensional array")

        if self.points.shape[1] != 3:
            raise ValueError("The points must be a N-by-3-dimensional array")

        if self.points.shape[0] < 3:
            raise ValueError("A valid polygon must have at least 3 points")

        self._parametric = None
        self._matrix = None
        self._shapely = None
        self._st_to_xyz_func = None

    def is_planar(self) -> bool:
        x0 = self.points
        x1 = np.roll(x0, 1, axis=0)
        x2 = np.roll(x1, 1, axis=0)
        v1 = x1 - x0
        v2 = x2 - x1
        normals = np.cross(v1, v2)

        return np.linalg.norm(np.diff(normals, axis=0)) < 1e-6

    @property
    def parametric(self):
        if self._parametric is None:
            # Plane calculation
            normal = np.cross(
                self.points[1, :] - self.points[0, :],
                self.points[2, :] - self.points[1, :],
            )
            normal /= np.linalg.norm(normal)  # Normalize
            a, b, c = normal
            d = -np.dot(np.array([a, b, c]), self.points[2, :])
            self._parametric = np.array([a, b, c, d])

        return self._parametric

    def cartesian(self, x, y, z):
        a, b, c, d = self.parametric
        return x * a + y * b + z * c + d

    def normal(self, x, y, z):
        return self.parametric[:3]

    def st_to_xyz(self, s, t):
        if self._st_to_xyz_func is None:
            a, b, c, d = self.parametric
            self._st_to_xyz_func = plane_st_to_xyz_function(a, b, c, d)

        return self._st_to_xyz_func(s, t)

    def contains(self, x, y, z):
        point = np.array([x, y, z])
        matrix = self.matrix()

        projected_polygon = self.points @ matrix
        projected_point = point @ matrix

        return shPolygon(projected_polygon[:, :2]).intersects(shPoint(projected_point))

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


if __name__ == "__main__":
    from raytracing.plotting import draw_polygon, new_3d_axes

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    p = Polygon(points)
    p.plot()
    p.show()

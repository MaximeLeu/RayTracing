from base import Geometry

from shapely.geometry import Polygon as shPolygon, Point as shPoint

import numba as nb


def plane_st_to_xyz_function(a, b, c, d):
    if a != 0:

        @nb.njit
        def __impl__(u, v):
            x = -(b * u + c * v + d) / a
            return x, u, v

    elif b != 0:

        @nb.njit
        def __impl__(u, v):
            y = -(a * u + c * v + d) / b
            return u, y, v

    elif c != 0:

        @nb.njit
        def __impl__(u, v):
            z = -(a * u + b * v + d) / c
            return u, v, z

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


class Polygon(Geometry):
    def __init__(self, points, **kwargs):
        super().__init__(**kwargs)

        self.points = np.asarray(points, dtype=float)

        if self.points.ndim != 2:
            raise ValueError("The points must be a  2-dimensional array")

        if self.shape[1] != 3:
            raise ValueError("The points must be a N-by-3-dimensional array")

        if self.shape[0] < 3:
            raise ValueError("A valid polygon must have at least 3 points")

        self._parametric = None
        self._matrix = None
        self._shapely = None

    @property
    def parametric(self):
        if self._parametric is None:
            # Plane calculation
            normal = np.cross(
                self.points[1, :] - self.points[0, :],
                self.points[2, :] - self.points[1, :],
            )
            normal /= np.dual.norm(normal)  # Normalize
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
            self._st_to_xyz_func = plane_st_to_xyz_func(a, b, c, d)

        return self._st_to_xyz_func(s, t)

    def contains(self, x, y, z):
        point = np.array([x, y, z])
        matrix = self.matrix()

        projected_polygon = self.points @ matrix
        projected_point = point @ matrix

        return shPolygon(projected_polygon[:, :2]).intersects(shPoint(projected_point))

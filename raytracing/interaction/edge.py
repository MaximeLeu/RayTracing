import numpy as np


class Edge(object):
    """
    Edge on which diffraction can occur.
    """

    def tangent(self, x, y, z):
        raise NotImplementedError

    def contains(self, x, y, z):
        raise NotImplementedError

    def s_to_xyz(self, s):
        raise NotImplementedError


class LinearEdge(Edge):
    def __init__(self, points, parent_surface_1, parent_surface_2=None):
        super().__init__()
        self.points = np.asarray(points, dtype=float).reshape(2, 3)
        self.parent_surface_1 = parent_surface_1
        self.parent_surface_2 = parent_surface_2
        self.__direction = self.points[1, :] - self.points[0, :]
        self.__tangent = self.__direction / np.linalg.norm(self.__direction)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return np.allclose(self.points, other.points) or np.allclose(
            self.points[::-1, :], other.points
        )

    def join(self, other):
        return LinearEdge(self.points, self.parent_surface_1, other.parent_surface_1)

    def tangent(self, point):
        return self.__tangent

    def contains(self, point, tol=1e-4):
        vec = point - self.points[0, :]
        dot = np.dot(vec, self.__direction) / np.dot(self.__direction, self.__direction)
        proj = point + dot * self.__direction
        return 0 <= dot <= 1  # and np.linalg.norm(point - proj) <= tol

    def s_to_xyz(self, s):
        return self.points[0, :] + self.__tangent * s

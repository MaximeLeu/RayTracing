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
        self.parent_surface_1 = parent_surface_2
        self.parent_surface_2 = parent_surface_2
        self.__tangent = self.points[1, :] - self.points[0, :]
        self.__tangent /= np.linalg.norm(self.__tangent)

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

    def contains(self, point, tol=1e-6):
        # return True
        vec = point - self.points[0, :]
        vec /= np.linalg.norm(vec)
        dot = np.dot(vec, self.__tangent)
        dist = np.linalg.norm(self.points[0, :] + dot * self.__tangent - point)
        return (0 <= dot <= 1) and dist < tol

    def s_to_xyz(self, s):
        return self.points[0, :] + self.__tangent * s

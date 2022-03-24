from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize

from ..interaction import Edge, Surface
from ..plotting import Plotable
from .base import bounding_box
from .path import Path


def assert_no_two_same_interactions(interact_list):
    it = iter(interact_list)
    n = next(it, None)

    while n is not None:
        p = next(it, None)
        if p == n:
            return False
        n = p

    return True


def st_to_xyz(points, interactions, ret, ret_normals=False):
    j = 0
    normals = []
    for i, inter in enumerate(interactions):
        k = 3 * i
        if isinstance(inter, Surface):
            ret[k : k + 3] = inter.st_to_xyz(*points[j : j + 2])
            normals.append(inter.normal(*points[j : j + 2]))
            j += 2
        else:
            ret[k : k + 3] = inter.s_to_xyz(points[j])
            normals.append(inter.tangent(points[j]))
            j += 1

    if ret_normals:
        return normals


def compute_path(start, end, interactions):

    if len(interactions) < 1:
        return Path(np.array([start.point, end.point]), start=start, end=end)

    nr = sum(1 for inter in interactions if isinstance(inter, Surface))
    nd = sum(1 for inter in interactions if isinstance(inter, Edge))

    def f(points, start, end, nr, nd, interactions):
        path = np.empty(3 * (nr + nd) + 6)
        path[:3] = start

        normals = st_to_xyz(points, interactions, path[3:-3], ret_normals=True)

        path[-3:] = end

        v = np.empty(nr * 3 + nd)

        j = 0
        for i, inter in enumerate(interactions):
            k = i * 3
            x0 = path[k + 0 : k + 3]
            x1 = path[k + 3 : k + 6]
            x2 = path[k + 6 : k + 9]
            v1 = x1 - x0
            v2 = x2 - x1

            if isinstance(inter, Surface):
                n = normals[i]
                v[j : j + 3] = np.linalg.norm(v1) * v2 - (
                    v1 - 2 * (np.dot(v1, n) * n)
                ) * np.linalg.norm(v2)

                j += 3
            else:
                e = normals[i]
                v[j] = np.dot(v1, e) * np.linalg.norm(v2) - np.dot(
                    v2, e
                ) * np.linalg.norm(v1)
                j += 1

        return np.linalg.norm(v)

    x0 = np.random.rand(nr * 2 + nd)

    sol = minimize(f, x0=x0, args=(start.point, end.point, nr, nd, interactions))

    path = np.empty(3 * (nd + nr))
    points = sol.x

    st_to_xyz(points, interactions, path)

    path = np.row_stack([start.point, path.reshape(-1, 3), end.point])

    return Path(path, start=start, end=end, res=sol, interact_list=interactions)


class Simulation(Plotable):
    def __init__(self, scene, TX, *RXS):
        super().__init__()

        self.scene = scene
        self.TX = TX
        self.RXS = RXS
        self.paths = []
        # self.domain = bounding_box([scene.domain, TX.domain, *[RX.domain for RX in RXS]])

    def compute_paths(
        self,
        max_interactions=1,
    ):
        self.paths.clear()

        interactions = [*self.scene.edges, *self.scene.surfaces]

        for i in range(0, max_interactions + 1):
            for interact_list in combinations_with_replacement(interactions, i):
                if not assert_no_two_same_interactions(interact_list):
                    continue

                for RX in self.RXS:
                    path = compute_path(self.TX, RX, interact_list)

                    if path.is_valid():
                        self.paths.append(path)

        return self.paths

    def plot(self):
        self.scene.on(self.ax).plot()
        for path in self.paths:
            path.on(self.ax).plot(color="m")

        self.TX.on(self.ax).plot(color="b")

        for RX in self.RXS:
            RX.on(self.ax).plot(color="r")

        return self.ax

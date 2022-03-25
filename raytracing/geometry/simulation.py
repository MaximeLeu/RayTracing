from itertools import combinations_with_replacement, tee
from math import comb

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm, trange

from ..interaction import Edge, Surface
from ..plotting import Plotable
from .base import bounding_box
from .path import Path


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def assert_no_two_same_interactions(interact_list):
    return not any(i1 != i2 for i1, i2 in pairwise(interact_list))


def assert_interactions_can_see(interact_list):
    return all(i1.can_see(i2) for i1, i2 in pairwise(interact_list))


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
        self.domain = bounding_box(
            [scene.domain, TX.domain] + [RX.domain for RX in RXS]
        )

    def compute_paths(
        self,
        max_interactions=3,
    ):
        self.paths.clear()
        surfaces = self.scene.surfaces
        edges = self.scene.edges

        interactions = [*surfaces, *edges]

        surface_indices = dict()

        n = len(interactions)
        m = len(surfaces)

        visibility = np.zeros((n + 2, n + 2), dtype=bool)

        for i, s1 in tqdm(enumerate(surfaces), total=m, leave=False):
            surface_indices[s1] = i
            for j in range(i + 1, m):
                s2 = surfaces[j]
                visibility[i, j] = s1.can_see(s2)

        visibility += visibility.T

        for k in range(m, n):
            edge = edges[k - m]
            i = surface_indices[edge.parent_surface_1]
            j = surface_indices[edge.parent_surface_2]
            visibility[k, :] = visibility[i, :] + visibility[j, :]
            visibility[k, i] = False
            visibility[k, j] = False

        visibility[-2, :] = True  # TX sees everyone
        visibility[:, -1] = True  # RX is seen by everyone

        G = nx.DiGraph(visibility)

        all_simple_paths = list(
            nx.all_simple_paths(G, source=n, target=n + 1, cutoff=max_interactions + 1)
        )

        for indices in tqdm(all_simple_paths):
            for RX in self.RXS:
                interact_list = [interactions[i] for i in indices[1:-1]]
                path = compute_path(self.TX, RX, interact_list)

                if path.is_valid() and not any(
                    surface.intersects(path.points) for surface in self.scene.surfaces
                ):
                    self.paths.append(path)

        """
        for i in trange(0, max_interactions + 1, leave=False):
            for interact_list in tqdm(
                combinations_with_replacement(interactions, i),
                total=comb(n + i - 1, i),
                leave=False,
            ):
                if not assert_no_two_same_interactions(interact_list):
                    continue

                for RX in self.RXS:
                    path = compute_path(self.TX, RX, interact_list)

                    if path.is_valid() and not any(
                        surface.intersects(path.points)
                        for surface in self.scene.surfaces
                    ):
                        self.paths.append(path)
        """
        return self.paths

    def plot(self):
        self.scene.on(self.ax).plot()
        for path in self.paths:
            path.on(self.ax).plot()

        self.TX.on(self.ax).plot(color="b")

        for RX in self.RXS:
            RX.on(self.ax).plot(color="r")

        self.set_limits(self.domain)

        return self.ax

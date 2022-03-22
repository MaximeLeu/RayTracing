"""
This file times several execution of the solver with the dummy geometry.
"""

import sys
import timeit

import numpy as np

import raytracing.geometry as geom
from raytracing import file_utils
from raytracing.electromagnetism import compute_field_from_solution
from raytracing.ray_tracing import RayTracingProblem


def f(r):
    # 1. Load data

    tx = np.array([5.0, 12.0, 5.0]).reshape(1, 3)
    rx = np.array([65.0, 12.0, 5.0]).reshape(1, 3)

    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, 0], [17, 24, 0]]))
    building_1 = geom.Building.by_polygon_and_height(square_1, 10)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, 0], [37, 2, 0]]))
    building_2 = geom.Building.by_polygon_and_height(square_2, 10)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, 0], [57, 24, 0]]))
    building_3 = geom.Building.by_polygon_and_height(square_3, 10)

    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [70, 24, 0]]))

    place = geom.OrientedPlace(
        geom.OrientedSurface(ground), [building_1, building_2, building_3]
    )

    problem = RayTracingProblem(tx, place, rx)
    problem.solve(r)

    compute_field_from_solution(problem)


if __name__ == "__main__":

    file_utils.chdir_to_file_dir(__file__)
    n = 10

    for r in range(0, 3):

        old = sys.stdout
        sys.stdout = None

        t = timeit.timeit(lambda: f(r), number=n)

        sys.stdout = old

        print("r=", r, ", t =", t / n, "s")

"""
This file shows you the correctness of the chained reflections algorithm.
"""

import random

import matplotlib.pyplot as plt
import numpy as np

import raytracing.geometry as geom
from raytracing import file_utils
from raytracing.plot_utils import (
    add_2d_text_at_point_3d_ax,
    add_points_to_3d_ax,
    add_text_at_point_3d_ax,
)

if __name__ == "__main__":

    file_utils.chdir_to_file_dir(__file__)

    A = np.array([0, 0, 0]).reshape(1, 3)
    B = np.array([6, 6, 6]).reshape(1, 3)

    cube = geom.Cube.by_point_and_side_length(A, 20)

    polygons = cube.get_polygons_iter()
    g = geom.polygons_sharp_edges_iter(polygons)

    ax = cube.plot3d(ret=True, alpha=0)
    add_points_to_3d_ax(ax, A)
    add_text_at_point_3d_ax(ax, A, "TX")
    add_points_to_3d_ax(ax, B)
    add_text_at_point_3d_ax(ax, B, "RX")
    cube.center_3d_plot(ax)
    cube.tight_3d_plot(ax)

    n = np.random.randint(1, 4)

    I = np.array([0.05, 0.95])

    add_2d_text_at_point_3d_ax(ax, I, f"Reflection on {n} wall(s) ")

    faces = cube.polygons
    random.shuffle(faces)

    X = [A]

    param = [faces[i].get_parametric() for i in range(n)]

    for i in range(n):
        faces[i].plot3d(facecolor="r", alpha=0.3, ax=ax, edgecolor="r")

    z, _ = geom.reflexion_points_from_origin_destination_and_planes(A, B, param)

    for point in z:
        X.append(point)
    X.append(B)
    for i in range(1, n + 2):
        x1, y1, z1 = X[i - 1].flat
        x2, y2, z2 = X[i].flat

        ax.plot([x1, x2], [y1, y2], [z1, z2], label=str(i))

    plt.legend()
    plt.show()

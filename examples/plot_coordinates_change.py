import radarcoverage.geometry as geom
from radarcoverage import plot_utils
import matplotlib.pyplot as plt
import numpy as np
from radarcoverage.plot_utils import add_points_to_3d_ax, add_text_at_point_3d_ax, add_2d_text_at_point_3d_ax
import random


if __name__ == '__main__':

    colors = ['b', 'r', 'g', 'm', 'y']

    # Cartesian coordinates
    points_A = np.array([
        [0, 0, 0],
        [1, 2, 3],
        [4, 6, 4]
    ])

    points_B = np.array([
        [-3, 2, -1],
        [-2, 5, 5],
        [1, 3, 4]
    ])

    polygon_A = geom.OrientedPolygon(points_A)
    polygon_B = geom.OrientedPolygon(points_B)
    polygon_C = polygon_B.translate(np.array([-1, -1, -1]))

    surface = geom.OrientedSurface([polygon_A, polygon_B, polygon_C])

    ax = surface.plot3d(ret=True)
    surface.center_3d_plot(ax)
    plot_utils.set_cartesian_axes_label(ax)

    for i, poly3d in enumerate(ax.collections):
        poly3d.set_facecolor(colors[i])

    # Polygon A coordinates

    centroid = polygon_A.get_centroid()
    matrix = polygon_A.get_matrix().T

    surface = surface.translate(-centroid).project(matrix)
    ax = surface.plot3d(ret=True)
    surface.center_3d_plot(ax)
    plot_utils.set_cartesian_axes_label(ax)

    for i, poly3d in enumerate(ax.collections):
        poly3d.set_facecolor(colors[i])

    # Spherical coordinates relative to polygon A

    surface = surface.project_on_spherical_coordinates().project_with_perspective_mapping(1)

    for polygon in surface.polygons:
        print(polygon.get_shapely())

    ax = surface.plot3d(ret=True)
    surface.center_3d_plot(ax)
    plot_utils.set_spherical_axes_label(ax)

    for i, poly3d in enumerate(ax.collections):
        poly3d.set_facecolor(colors[i])


    plt.show()
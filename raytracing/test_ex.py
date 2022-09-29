# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:33:28 2022

@author: maxime
"""

from ray_tracing import RayTracingProblem
import raytracing.geometry as geom
import numpy as np
from raytracing import plot_utils
import matplotlib.pyplot as plt



if __name__ == '__main__':
    tx = np.array([5., 12., 5.]).reshape(1, 3)
    rx = np.array([65., 12., 5.]).reshape(1, 3)

    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, 0], [17, 24, 0]]))
    building_1 = geom.Building.by_polygon_and_height(square_1, 10)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, 0], [37, 2, 0]]))
    building_2 = geom.Building.by_polygon_and_height(square_2, 10)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, 0], [57, 24, 0]]))
    building_3 = geom.Building.by_polygon_and_height(square_3, 10)

    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [70, 24, 0]]))

    place = geom.OrientedPlace(geom.OrientedSurface(ground), [building_1, building_2, building_3])

    # Adding receivers to place
    place.add_set_of_points(rx)
    ax = place.plot3d(ret=True)
    #place.center_3d_plot(ax)

    ax=plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")
    #ax = problem.plot3d(ax=ax)
    plt.legend()

        # This plot is here to check that you geometry is correct.
    plt.show()


    RayTracingProblem(tx, place)
    problem=RayTracingProblem(tx, place)


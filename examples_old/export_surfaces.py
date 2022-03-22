"""
This file shows you how to plot the geometry in 3D.
"""

import matplotlib.pyplot as plt
import numpy as np

import raytracing.geometry as geom
from raytracing import file_utils

if __name__ == "__main__":

    file_utils.chdir_to_file_dir(__file__)

    filename = "../data/small.geojson"
    place = geom.generate_place_from_rooftops_file(filename)

    matrix = place.get_visibility_matrix(strict=True)
    filename = "/home/jeertmans/repositories/DiffeRT/DiffeRT-research/latex/03-vtc-spring-2022/visibility.mat"
    with open(filename, "w") as f:
        np.savetxt(filename, matrix)
    np.savetxt(
        "/home/jeertmans/repositories/DiffeRT/DiffeRT-research/latex/03-vtc-spring-2022/parametrics.mat",
        np.row_stack([poly.get_parametric() for poly in place.get_polygons_list()]),
    )

    ax = place.plot3d(ret=True)

    place.center_3d_plot(ax)

    plt.show()

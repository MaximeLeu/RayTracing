"""
This file shows you how to plot the geometry in 3D.
"""

import raytracing.geometry as geom
from raytracing import file_utils
import matplotlib.pyplot as plt

if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    filename = '../data/ny.geojson'
    place = geom.generate_place_from_rooftops_file(filename)

    ax = place.plot3d(ret=True)

    place.center_3d_plot(ax)

    plt.show()

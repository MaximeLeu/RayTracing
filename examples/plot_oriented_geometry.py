"""
This tutorial shows you how to plot the orientation of the polygons in your geometry.
"""

import raytracing.geometry as geom
from raytracing import file_utils
import matplotlib.pyplot as plt

if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    filename = '../data/small.geojson'
    place = geom.generate_place_from_rooftops_file(filename)
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    place.center_3d_plot(ax)
    ax = place.plot3d(ax=ax,ret=True, poly_kwargs=dict(orientation=True, normal=True))
    plt.show()

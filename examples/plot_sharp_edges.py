import radarcoverage.geometry as geom
from radarcoverage import file_utils
import matplotlib.pyplot as plt
import numpy as np
import radarcoverage.plot_utils as plot_utils


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    place = geom.generate_place_from_rooftops_file('../data/small.geojson')

    #place = geom.Cube.by_point_and_side_length(np.array([0, 0, 0]), 10)
    place.show_sharp_edges_animation()

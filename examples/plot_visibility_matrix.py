import radarcoverage.geometry as geom
from radarcoverage import file_utils
import matplotlib.pyplot as plt
import numpy as np
import radarcoverage.plot_utils as plot_utils


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    place = geom.generate_place_from_rooftops_file('../data/ny.geojson')

    place.show_visibility_matrix_animation(strict=True)

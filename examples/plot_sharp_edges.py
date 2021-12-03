"""
This file shows you how to verify visually the sharp edges in your geometry.
"""

import raytracing.geometry as geom
from raytracing import file_utils


if __name__ == "__main__":

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    place = geom.generate_place_from_rooftops_file("../data/small.geojson")

    place.show_sharp_edges_animation()

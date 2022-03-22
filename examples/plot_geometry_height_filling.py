"""
This file is here to generate images for the tutorial.
"""

import matplotlib.pyplot as plt

import raytracing.geometry as geom
from raytracing import file_utils

if __name__ == "__main__":

    file_utils.chdir_to_file_dir(__file__)

    filename = "../docs/source/tutorials/data/export_modified.geojson"
    place = geom.generate_place_from_rooftops_file(
        filename, drop_missing_heights=False, default_height=30
    )
    import numpy as np

    # Place cube at center, height z = 20m and length L = 10m
    cube = geom.Cube.by_point_and_side_length(np.array([0, 0, 20]), 10)

    # Because I'm lazy, I take the bottom surface of the cube (always index 1)
    # as a square base

    square = cube.polygons[1]  # At [0, 0, 15]

    # Building
    building_base = square.translate(np.array([-60, 20, -15]))  # At [-60, 20, 0]
    building = geom.Building.by_polygon2d_and_height(building_base, 20)

    # Pyramid
    pyramid_base = square.translate(np.array([60, -20, -15]))

    # Will place the top of the pyramid in the middle of the base
    point = pyramid_base.get_centroid() + np.array([0, 0, 40])

    # Let's do fancy rotation

    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    pyramid_base = pyramid_base.project(matrix.T, around_point=point)
    pyramid = geom.Pyramid.by_point_and_polygon(point, pyramid_base)

    place.polyhedra.extend([cube, building, pyramid])

    ax = place.plot3d(ret=True, poly_kwargs=dict(orientation=True, normal=True))

    place.center_3d_plot(ax)

    plt.show()

    place.to_json("lln.json")

    # place.get_sharp_edges(min_angle=0)  # Only if you want to override default value
    place.show_visibility_matrix_animation(strict=False)

    plt.show()

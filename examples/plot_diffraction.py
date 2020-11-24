"""
This file shows you the correctness of the diffraction algorithm.
Warning: not all diffractions are computed here.
"""

import raytracing.geometry as geom
from raytracing import file_utils
import matplotlib.pyplot as plt
import numpy as np
from raytracing import plot_utils
if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    A = np.array([-0.5, -5, -1]).reshape(1, 3)
    B = np.array([-2.5, -5, 1]).reshape(1, 3)

    frame = geom.Cube.by_point_and_side_length(A, 10)

    cube = geom.Cube.by_point_and_side_length(0*A, 5)

    face = cube.polygons[5]
    vertice = face.points[:2, :]

    screen = face.translate(np.array([0, -5, 0]))

    ax = cube.plot3d(ret=True, alpha=0.1)
    screen.plot3d(ax=ax)
    plot_utils.add_points_to_3d_ax(ax, A)
    plot_utils.add_text_at_point_3d_ax(ax, A, 'TX')
    plot_utils.add_points_to_3d_ax(ax, B)
    plot_utils.add_text_at_point_3d_ax(ax, B, 'RX')
    plot_utils.add_line_to_3d_ax(ax, vertice, color='r', lw=3)
    frame.center_3d_plot(ax)

    diff_point_1, _ = geom.diffraction_point_from_origin_destination_and_edge(A, B, vertice)
    diff_point_2, _ = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(A, B, [screen.get_parametric()], vertice)

    diff_line_1 = np.row_stack([A, diff_point_1, B])
    diff_line_2 = np.row_stack([A, diff_point_2, B])
    plot_utils.add_line_to_3d_ax(ax, diff_line_1, color='b', linestyle='--', label='single diffraction')
    plot_utils.add_line_to_3d_ax(ax, diff_line_2, color='r', linestyle='--', label='reflection + diffraction')

    ref_point, sol = geom.reflexion_points_from_origin_destination_and_planes(A, B, [screen.get_parametric()])
    ref_line = np.row_stack([A, ref_point, B])
    plot_utils.add_line_to_3d_ax(ax, ref_line, alpha=0.5, label='single reflection')

    ref_points, sol = geom.reflexion_points_from_origin_destination_and_planes(A, B, [screen.get_parametric(), face.get_parametric()])


    ref_line = np.row_stack([A, ref_points, B])
    plot_utils.add_line_to_3d_ax(ax, ref_line, alpha=0.5, label='double reflection')

    plt.legend()

    plt.show()

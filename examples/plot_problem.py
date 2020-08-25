import radarcoverage.geometry as geom
from radarcoverage import file_utils
import matplotlib.pyplot as plt
import numpy as np
import radarcoverage.plot_utils as plot_utils


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    place = geom.generate_place_from_rooftops_file('../data/small.geojson')


    # 2. Create TX and RX

    domain = place.get_domain()
    ground_center = place.get_centroid()

    tx = ground_center + [-50, 5, 1]
    rx = ground_center + [35, 5, 5]
    tx = tx.reshape(1, 3)
    rx = rx.reshape(1, 3)

    # 2.1 Create a cube around TX

    distance = 5
    cube = geom.Cube.by_point_and_side_length(tx, 2 * distance)

    # 2.1.1 Rotate this cube around its center
    from scipy.spatial.transform import Rotation as R

    rot2 = R.from_euler('xyz', [0, 10, -10], degrees=True).as_matrix()

    cube = cube.project(rot2, around_point=tx)

    # 2.2 Place TX and RX in the 'place'
    #place.add_set_of_points(tx)
    place.add_set_of_points(rx)

    # 2.3 Translate the geometry around TX

    place = place.translate(-tx)
    cube = cube.translate(-tx)
    polygons = place.get_polygons_list()

    print(place.get_visibility_matrix()[0, :])
    # 3. Plot the whole geometry
    ax1 = place.plot3d(ret=True)
    cube.plot3d(ax=ax1)

    rx -= tx
    tx -= tx
    plot_utils.add_vector_to_3d_ax(ax1, tx, rx - tx)

    points = np.row_stack([tx, rx])

    print('Place intersects line:', place.obstructs_line_path(points))

    place.center_3d_plot(ax1)

    # 3.1 Picking one face of the cube as the screen and coloring it
    screen = cube.polygons[2]

    screen.plot3d(facecolor='g', alpha=0.5, ax=ax1, orientation=True, normal=True)
    # 4. Create the screen on which geometry will be projected

    # 5. First, changing geometry coordinates to match screen's orientation
    matrix = screen.get_matrix().T

    projected_place = place.project(matrix)
    screen = screen.project(matrix)

    ax2 = projected_place.plot3d(ret=True)
    cube.plot3d(ax=ax2)
    screen.plot3d(facecolor='g', ax=ax2)

    projected_place.center_3d_plot(ax2)


    def filter_func(polygon):
        return np.dot(polygon.get_normal(), screen.get_normal()) < np.arccos(np.pi/4) and\
               geom.any_point_above(polygon.points, 0, axis=2)

    poly_matching = projected_place.get_polygons_matching(filter_func, polygons)

    screen2d = screen.get_shapely()

    print('Screen in 2D:', screen2d)

    visible_polygons = list()

    for poly_changed, poly_original in poly_matching:
        poly_changed = poly_changed.project_with_perspective_mapping(focal_distance=distance)
        shp = poly_changed.get_shapely()
        if shp.intersects(screen2d):
            parametric = poly_original.get_parametric()
            point = geom.reflexion_point_from_origin_destination_and_planes(tx, rx, [parametric])

            if poly_original.contains_point(point[0]):

                if not place.obstructs_line_path(np.row_stack([tx, point[0]])):
                    if not place.obstructs_line_path(np.row_stack([point[0], rx])):

                        plot_utils.add_points_to_3d_ax(ax1, point)

                        line = np.row_stack([tx, point[0], rx])

                        plot_utils.add_line_to_3d_ax(ax1, line)


    #print(len(list(poly)))
    # 6. Perspective mapping on z direction
    projected_place = projected_place.project_with_perspective_mapping(focal_distance=distance)

    screen = screen.project_with_perspective_mapping(focal_distance=distance)

    rot = R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()

    ax3 = projected_place.project(rot).plot2d(ret=True, poly_kwargs=dict(alpha=0.5))

    screen.project(rot).plot2d(ax=ax3, facecolor='g', alpha=0.4)
    plt.axis('equal')

    plt.show()

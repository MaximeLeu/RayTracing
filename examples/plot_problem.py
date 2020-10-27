import raytracing.geometry as geom
from raytracing import file_utils
import numpy as np
from time import time
from raytracing.ray_tracing import RayTracingProblem
import matplotlib.pyplot as plt


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    geometry = 'dummy'

    if geometry == 'small':
        place = geom.generate_place_from_rooftops_file('../data/small.geojson')

        # 2. Create TX and RX

        domain = place.get_domain()
        ground_center = place.get_centroid()

        tx = ground_center + [-50, 5, 1]
        rx = ground_center + np.array([
            [35, 5, 5],
            [35, -5, 5],
            [10, -3, -5]
        ])
        rx = rx[2, :]
        tx = tx.reshape(-1, 3)
        rx = rx.reshape(-1, 3)

        # 2.1 Create a cube around TX

        distance = 5
        cube = geom.Cube.by_point_and_side_length(tx, 2 * distance)
        # 2.1.1 Rotate this cube around its center
        from scipy.spatial.transform import Rotation as R

        rot2 = R.from_euler('xyz', [0, 10, -10], degrees=True).as_matrix()

        cube = cube.project(rot2, around_point=tx)
        screen = cube.polygons[2]
    elif geometry == 'dummy':
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

        cube = geom.Cube.by_point_and_side_length(tx, 5)
        screen = cube.polygons[2]

    # place.show_visibility_matrix_animation(True)

    t = time()
    problem = RayTracingProblem(tx, screen, place, rx)
    print(f'Took {time() - t:.4f} seconds to initialize and precompute problem.')

    t = time()
    problem.solve(3)
    print(f'Took {time() - t:.4f} seconds to solve problem.')
    problem.plot3d()

    problem.save('../data/problem.json')

    from raytracing.electromagnetism import compute_field_from_solution

    compute_field_from_solution(problem)

    plt.show()


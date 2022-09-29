"""
This file show how to use the Ray Tracing solver.
"""


import raytracing.geometry as geom
from raytracing import file_utils
from raytracing import plot_utils
import numpy as np
from time import time
from raytracing.ray_tracing import RayTracingProblem
import matplotlib.pyplot as plt


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    geometry = 'my_geometry'  # You can change to use another geometry

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

    elif geometry == 'my_geometry':
        filename="../data/TutorialTest/sciences.geojson"
        
        geom.preprocess_geojson(filename)
        filename=geom.sample_geojson(filename,nBuildings=10)
        place = geom.generate_place_from_rooftops_file(filename)

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
        
        
        
        
        
    # Adding receivers to place
    place.add_set_of_points(rx)
    ax = place.plot3d(ret=True)
    place.center_3d_plot(ax)

    plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")

    plt.legend()

    # This plot is here to check that you geometry is correct.
    plt.show()

    t = time()
    problem = RayTracingProblem(tx, place)
    print(f'Took {time() - t:.4f} seconds to initialize and precompute problem.')

    t = time()
    problem.solve(max_order=2)
    print(f'Took {time() - t:.4f} seconds to solve problem.')
    fig = plt.figure()
    ax = fig.gca(projection="3d")


#pour créer une sorte de heatmap en dessous du plot. Inutile.
    # surface = place.surface
    # domain = surface.get_domain()

    # x = np.linspace(domain[0,0], domain[1,0], 100)
    # y = np.linspace(domain[0,1], domain[1,1], 100)

    # X, Y = np.meshgrid(x, y)
    # Z = np.cos((X+50)/120)*np.cos((Y)/40)
    # Z += 0.05 * np.random.random(Z.shape)
    # C = X + 1j * Y
    # for dc in [10 + 10j, 30 + 30j, 20 - 30j]:
    #     A = C - dc
    #     Z += 0.2 * np.exp(-np.real(A * np.conj(A)) / 20)

    # it = np.nditer([X, Y, Z], [], [['readonly'], ['readonly'], ['writeonly']])

    # surfaces = [place.polyhedra[0].get_polygons_list()[0], place.polyhedra[1].get_polygons_list()[0], place.polyhedra[2].get_polygons_list()[0]]
    # for a, b, c in it:
    #     point = np.array([a, b, c])
    #     for surf in surfaces:
    #         if surf.contains_point(point):
    #             np.add(0, 0, out=c)
    #             break

    #ax.contourf(X, Y, Z, zdir="z", offset=-30)
    
    ax = problem.plot3d(ax=ax)

    problem.save('../data/problem.json')

    from raytracing.electromagnetism import compute_field_from_solution

    compute_field_from_solution(problem, '../data/electromagnetism.json')

    plt.show()


from radarcoverage import geometry as geom
from radarcoverage import plot_utils

from time import time

import numpy as np

from collections import defaultdict


class RayTracingProblem:

    def __init__(self, emitter, emitter_screen, place, receivers=None):
        self.emitter = emitter
        self.emitter_screen = emitter_screen
        self.place = place
        self.polygons = np.array(place.get_polygons_list())

        if receivers is not None:
            self.place.add_set_of_points(receivers)

        self.receivers = self.place.set_of_points

        self.visibility_matrix = None
        self.distance_to_screen = None
        self.emitter_visibility = None
        self.sharp_edges = None
        self.los = list()  # Line of sight
        self.reflections = defaultdict(list)
        self.diffractions = defaultdict(list)
        self.precompute()

    def precompute(self):
        self.visibility_matrix = self.place.get_visibility_matrix(strict=False)
        self.distance_to_screen = self.emitter_screen.distance_to_point(self.emitter)
        self.emitter_visibility = geom.polygon_visibility_vector(
            self.emitter_screen, self.polygons, strict=True
        )
        self.sharp_edges = self.place.get_sharp_edges()

        screen_matrix = self.emitter_screen.get_matrix().T

        projected_screen = self.emitter_screen.translate(-self.emitter).project(screen_matrix)
        screen_shapely = projected_screen.get_shapely()

        for i, is_visible in enumerate(self.emitter_visibility):
            if is_visible:
                projected_polygon = self.polygons[i].translate(-self.emitter).project(screen_matrix)
                projected_with_perspective_polygon = projected_polygon.project_with_perspective_mapping(
                    focal_distance=self.distance_to_screen
                )
                if not screen_shapely.intersects(projected_with_perspective_polygon.get_shapely()):
                    self.emitter_visibility[i] = False

        visible_polygons = self.polygons[self.emitter_visibility]

        for receiver in self.receivers:
            projected_receiver = geom.project_points(receiver - self.emitter, screen_matrix)
            projected_with_perspective_receiver = geom.project_points_with_perspective_mapping(
                projected_receiver,
                focal_distance=self.distance_to_screen
            )
            point = geom.shPoint(projected_with_perspective_receiver.reshape(-1))
            if screen_shapely.intersects(point):
                line = np.row_stack([self.emitter, receiver])
                if not geom.polygons_obstruct_line_path(visible_polygons, line):
                    self.los.append(line)

    def get_visible_polygons_indices(self, index):
        indices = self.visibility_matrix[index, :]
        return np.where(indices)[0]

    def check_reflections(self, lines, polygons_indices):
        for i, index in enumerate(polygons_indices):
            if not self.polygons[index].contains_point(lines[i + 1:i + 2, :], check_in_plane=True):
                return False

        if geom.polygons_obstruct_line_path(self.polygons, lines[:2, :]):
            return False

        for i, _ in enumerate(polygons_indices):
            if geom.polygons_obstruct_line_path(self.polygons, lines[i + 1:i + 3, :]):
                return False

        return True

    def check_reflections_and_diffraction(self, lines, polygons_indices, edge):
        return self.check_reflections(lines, polygons_indices) and \
               geom.point_on_edge_(lines[-2, :], edge) and \
               not geom.polygons_obstruct_line_path(self.polygons, lines[-2:, :])

    def solve(self, max_order=2):

        visible_polygons = self.polygons[self.emitter_visibility]
        emitter = self.emitter
        receivers = self.receivers
        indices = np.where(self.emitter_visibility)[0]

        # Only reflections
        def recursive_reflections(polygons_indices, order):
            planes_parametric = [self.polygons[index].get_parametric() for index in polygons_indices]

            for receiver in receivers:
                points, sol = geom.reflexion_points_from_origin_destination_and_planes(emitter, receiver, planes_parametric)

                if not sol.success:
                    continue

                lines = np.row_stack([emitter, points, receiver])
                if self.check_reflections(lines, polygons_indices):
                    self.reflections[order].append(lines)

            if order == max_order:
                return
            else:
                index = polygons_indices[-1]
                indices = self.get_visible_polygons_indices(index)
                for i in indices:
                    recursive_reflections(polygons_indices + [i], order=order + 1)

        for index in indices:
            recursive_reflections([index], 1)

        # Reflections and 1 diffraction
        for (i, j), edge in self.sharp_edges.items():
            if self.emitter_visibility[i] or self.emitter_visibility[j]:
                for receiver in receivers:
                    points, sol = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(emitter, receiver, [], edge)

                    if not sol.success:
                        continue

                    lines = np.row_stack([emitter, points, receiver])
                    if self.check_reflections_and_diffraction(lines, [], edge):
                        self.diffractions[1].append(lines)

    def plot3d(self, ax=None, ret=False):
        ax = plot_utils.get_3d_plot_ax(ax)

        self.place.plot3d(ax=ax, points_kwargs=dict(color='k', s=20))

        plot_utils.add_points_to_3d_ax(ax, self.emitter, color='r', s=20)

        self.emitter_screen.plot3d(ax=ax, facecolor='g', alpha=0.5)

        for line in self.los:
            plot_utils.add_line_to_3d_ax(ax, line, color='b')

        colors = {
            1: 'g',
            2: 'm',
            3: 'y',
            4: 'o'
        }

        for order, lines in self.reflections.items():
            color = colors[order]
            for line in lines:
                plot_utils.add_line_to_3d_ax(ax, line, color=color)
                plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

        for order, lines in self.diffractions.items():
            color = colors[order]
            for line in lines:
                plot_utils.add_line_to_3d_ax(ax, line, color=color, linestyle='--')
                plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

        self.place.center_3d_plot(ax)

        if ret:
            return ax


if __name__ == '__main__':

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
    #rx = rx[1, :]
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

    t = time()
    problem = RayTracingProblem(tx, screen, place, rx)
    print(f'Took {time()-t:.4f} seconds to initialize and precompute problem.')

    t = time()
    problem.solve(2)
    print(f'Took {time()-t:.4f} seconds to solve problem.')
    problem.plot3d()

    import matplotlib.pyplot as plt

    plt.show()
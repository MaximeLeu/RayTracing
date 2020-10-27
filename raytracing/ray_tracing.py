from raytracing import geometry as geom
from raytracing import plot_utils
from raytracing import file_utils

import numpy as np

from collections import defaultdict
from tqdm import tqdm


class RayTracingProblem:
    """
    A ray tracing problem instance offers tools to find paths between a given emitter and receivers.
    A path may be direct or indirect, as a result of reflection and/or diffraction.

    The paths are found regardless of the type of rays (sound wave, E.M. wave, light wave, ...).

    :param emitter: the emitter point
    :type emitter: numpy.array *size=(3)*
    :param emitter_screen: the polygon screen, as all the emitted rays must intersect this screen
    :type emitter_screen: raytracing.OrientedPolygon
    :param place: the place of the problem, which should contain buildings and a ground surface
    :type place: raytracing.OrientedPlace
    :param receivers: the receiver points but will also take the points contained in the place as receivers
    :type receivers: numpy.array *shape=(N, 3)*
    """
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
        n = self.receivers.shape[0]
        self.los = {
            r: list()  # Lines of sight
            for r in range(n)
        }
        self.reflections = {
            r: defaultdict(list)
            for r in range(n)
        }
        self.diffractions = {
            r: defaultdict(list)
            for r in range(n)
        }
        self.precompute()

    def precompute(self):
        """
        Pre-computes order-of-path independent results such as:
        - visibility matrices
        - sharp edges
        - lines of sight (a.k.a. direct paths)
        """
        self.visibility_matrix = self.place.get_visibility_matrix(strict=False)
        self.distance_to_screen = self.emitter_screen.distance_to_point(self.emitter)
        self.emitter_visibility = geom.polygon_visibility_vector(
            self.emitter_screen, self.polygons, strict=False
        )
        self.sharp_edges = self.place.get_sharp_edges()

        screen_matrix = self.emitter_screen.get_matrix().T

        projected_screen = self.emitter_screen.translate(-self.emitter).project(screen_matrix)
        screen_shapely = projected_screen.get_shapely()

        """
        for i, is_visible in enumerate(self.emitter_visibility):
            if is_visible:
                projected_polygon = self.polygons[i].translate(-self.emitter).project(screen_matrix)
                projected_with_perspective_polygon = projected_polygon.project_with_perspective_mapping(
                    focal_distance=self.distance_to_screen
                )
                if not screen_shapely.intersects(projected_with_perspective_polygon.get_shapely()):
                    self.emitter_visibility[i] = False
        """

        visible_polygons = self.polygons[self.emitter_visibility]

        for r, receiver in enumerate(self.receivers):
            projected_receiver = geom.project_points(receiver - self.emitter, screen_matrix)
            projected_with_perspective_receiver = geom.project_points_with_perspective_mapping(
                projected_receiver,
                focal_distance=self.distance_to_screen
            )
            point = geom.shPoint(projected_with_perspective_receiver.reshape(-1))
            if screen_shapely.intersects(point):
                line = np.row_stack([self.emitter, receiver])
                if not geom.polygons_obstruct_line_path(visible_polygons, line):
                    self.los[r].append(line)

    def get_visible_polygons_indices(self, index):
        indices = self.visibility_matrix[index, :]
        return np.where(indices)[0]

    def check_reflections(self, lines, polygons_indices):
        """
        Checks whether a reflection path is valid. It is valid if it satisfies 2 conditions:
        1. Each point is contained in the polygon it should be in
        2. No polygon obstructs the line path

        :return: True if reflection path is valid
        :rtype: bool
        """
        for i, index in enumerate(polygons_indices):
            if not self.polygons[index].contains_point(lines[i + 1, :], check_in_plane=True):
                return False

        if geom.polygons_obstruct_line_path(self.polygons, lines[:2, :]):
            return False

        for i, _ in enumerate(polygons_indices):
            if geom.polygons_obstruct_line_path(self.polygons, lines[i + 1:i + 3, :]):
                return False

        return True

    def check_reflections_and_diffraction(self, lines, polygons_indices, edge):
        return self.check_reflections(lines[:-1, :], polygons_indices) and \
               geom.point_on_edge_(lines[-2, :], edge) and \
               not geom.polygons_obstruct_line_path(self.polygons, lines[-2:, :])

    def solve(self, max_order=2):
        emitter = self.emitter
        receivers = self.receivers
        indices = np.where(self.emitter_visibility)[0]

        # Only reflections
        def recursive_reflections(polygons_indices, order):
            planes_parametric = [self.polygons[index].get_parametric() for index in polygons_indices]

            for r, receiver in enumerate(receivers):
                points, sol = geom.reflexion_points_from_origin_destination_and_planes(emitter, receiver, planes_parametric)

                if not sol.success:
                    continue

                lines = np.row_stack([emitter, points, receiver])
                if self.check_reflections(lines, polygons_indices):
                    self.reflections[r][order].append((lines, polygons_indices))

            if order == max_order:
                return
            else:
                index = polygons_indices[-1]
                indices = self.get_visible_polygons_indices(index)
                for i in indices:
                    recursive_reflections(polygons_indices + [i], order=order + 1)

        if max_order >= 1:
            print('Iterating through all n reflect.')
            for index in tqdm(indices):
                recursive_reflections([index], 1)

        if max_order < 1:
            return

        # Reflections and 1 diffraction
        print('Iterating through all 1 diff.')
        for (i, j), edge in tqdm(self.sharp_edges.items()):
            if self.emitter_visibility[i] or self.emitter_visibility[j]:
                for r, receiver in enumerate(receivers):
                    points, sol = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(
                        emitter, receiver, [], edge
                    )

                    if not sol.success:
                        continue

                    lines = np.row_stack([emitter, points, receiver])
                    if self.check_reflections_and_diffraction(lines, [], edge):
                        self.diffractions[r][1].append((lines, [], (i, j), edge))

        if max_order < 2:
            return

        def recursive_reflections_and_diffraction(polygons_indices, order):
            planes_parametric = [self.polygons[index].get_parametric() for index in polygons_indices]

            last_index = polygons_indices[-1]

            visible_polygons_indices = self.get_visible_polygons_indices(last_index)

            for edge_polygons, edge in self.sharp_edges[(*visible_polygons_indices, ...)]:
                for r, receiver in enumerate(receivers):
                    points, sol = geom.reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(emitter, receiver, planes_parametric, edge)

                    if not sol.success:
                        continue

                    lines = np.row_stack([emitter, points, receiver])
                    if self.check_reflections_and_diffraction(lines, polygons_indices, edge):
                        self.diffractions[r][order].append((lines, polygons_indices, edge_polygons, edge))

            if order == max_order:
                return
            else:
                index = polygons_indices[-1]
                indices = self.get_visible_polygons_indices(index)
                for i in indices:
                    recursive_reflections_and_diffraction(polygons_indices + [i], order=order + 1)

        print('Iterating through all n-1 reflect. and 1 diff.')
        for index in tqdm(indices):
            recursive_reflections_and_diffraction([index], 2)

    def save(self, filename):
        data = {
            'place': self.place.to_json(),
            'emitter': self.emitter,
            'los': self.los,
            'reflections': self.reflections,
            'diffractions': self.diffractions
        }

        file_utils.json_save(filename, data, cls=geom.OrientedGeometryEncoder)

    def plot3d(self, ax=None, ret=False):
        ax = plot_utils.get_3d_plot_ax(ax)

        self.place.plot3d(ax=ax, points_kwargs=dict(color='k', s=20))

        plot_utils.add_points_to_3d_ax(ax, self.emitter, color='r', s=20)
        plot_utils.add_text_at_point_3d_ax(ax, self.emitter, 'TX')
        self.emitter_screen.plot3d(ax=ax, facecolor='g', alpha=0.5)

        first = True
        handles = []
        labels = []

        for r, _ in enumerate(self.receivers):

            for line in self.los[r]:
                line3D, = plot_utils.add_line_to_3d_ax(ax, line, color='b')
                if first:
                    handles.append(line3D)
                    labels.append('LOS')
                    first = False

            colors = {
                1: 'g',
                2: 'm',
                3: 'y',
                4: 'r'
            }

            for order, lines in self.reflections[r].items():
                first = True
                color = colors[order]
                for line, _ in lines:
                    line3D, = plot_utils.add_line_to_3d_ax(ax, line, color=color)
                    plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

                    if first:
                        handles.append(line3D)
                        labels.append(f'{order} reflect.')
                        first = False

            for order, lines in self.diffractions[r].items():
                first = True
                color = colors[order]
                for line, _, _, _ in lines:
                    line3D, = plot_utils.add_line_to_3d_ax(ax, line, color=color, linestyle='--')
                    plot_utils.add_points_to_3d_ax(ax, line[1:order+1, :], color=color)

                    if first:
                        handles.append(line3D)
                        labels.append(f'{order-1} reflect. and 1 diff.')
                        first = False

        self.place.center_3d_plot(ax)
        ax.legend(handles, labels)

        if ret:
            return ax
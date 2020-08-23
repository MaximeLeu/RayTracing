# Plotting libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
from radarcoverage import plot_utils

# Numerical libraries
import numpy as np

# Geometry libraries
from shapely.geometry import Polygon as shPolygon
import geopandas as gpd

# Utils
import itertools


def enclosed_area(points):
    """
    Returns the enclosed area of the polygon described by the points.
    The polygon is projected on the z=0 plane.
    If results is negative, it means that the curve orientation is ccw.

    :param points: the points of the polygon
    :type points: numpy.ndarray *shape=(N, 2 or 3)*
    :return: the enclosed area
    :rtype: float
    """
    # From:
    # https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    x = points[:, 0]
    y = points[:, 1]
    s = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
    s += (x[0] - x[-1]) * (y[0] + y[-1])
    return s / 2


def is_ccw(points):
    """
    Returns if the curve described by the points is oriented ccw or not.
    The curve is projected on the z=0 plane.

    :param points: the points of the polygon
    :type points: numpy.ndarray *shape=(N, 2 or 3)*
    :return: True if the curve is oriented ccw
    :rtype: bool
    """
    return enclosed_area(points) < 0


def parse_3d_axis(axis):
    """
    Parses an axis to a valid axis in 3D geometry, i.e. 0, 1 or 2 (resp. to x, y, or z).

    :param axis: axis to be parsed
    :type axis: int (-3 to 2) or str (x, y or z)
    :return: the axis
    :rtype int
    """
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == 'x':
            axis = 0
        elif axis == 'y':
            axis = 1
        elif axis == 'z':
            axis = 2
        else:
            raise ValueError(f'Cannot parse {axis} to a valid axis.')
    else:
        axis = (int(axis) + 3) % 3

    return axis


def reflexion_on_plane(incidents, normal):
    """
    Return the reflexion of incident vector on a plane with given normal.
    See details: https://en.wikipedia.org/wiki/Reflection_(mathematics)

    :param incidents: incident vectors
    :type incidents: numpy.ndarray *shape=(N, 3)*
    :param normal: normal vector to the plane
    :type normal: numpy.ndarray *size=3*
    :return: the reflected vector(s)
    :rtype: numpy.ndarray *shape(N, 3)*
    """
    normal = normal.reshape((1, 3))
    incidents = incidents.reshape((-1, 3))
    den = normal @ normal.T
    return incidents - (incidents @ normal.T) @ ((2 / den) * normal)  # Order of operation minimizes the # of op.


def translate_points(points, vector):
    """
    Translates points using a vector as displacement.

    :param points: the points to translate
    :type points: numpy.ndarray *shape=(N, 3)*
    :param vector: the displacement vector
    :type vector: numpy.ndarray *size=3*
    :return: the new points
    :rtype: numpy.ndarray *shape=(N, 3)*
    """
    return points.reshape(-1, 3) + vector.reshape(1, 3)


def project_points(points, matrix, around_point=None):
    """
    Projects points on different axes given by matrix columns.

    :param points: the points to be projected
    :type points: numpy.ndarray *shape=(N, 3)*
    :param matrix: the matrix to project all the geometry
    :type matrix: numpy.ndarray *shape=(3, 3)*
    :param around_point: if present, will apply the projection around this point
    :type around_point: numpy.ndarray *size=3*
    :return: the projected points
    :rtype: numpy.ndarray *shape=(N, 3)*
    """
    if around_point is not None:
        around_point = around_point.reshape(1, 3)
        return ((points - around_point) @ matrix.T) + around_point
    else:
        return points @ matrix.T


def project_points_with_perspective_mapping(points, focal_distance=1, axis=2):
    """
    Projects points with a perspective using similar triangles rule.
    The screen on which the points are projected is at a given distance from the origin.

    :param points: the points to be projected
    :type points: numpy.ndarray *shape=(N, 3)*
    :param focal_distance: the distance to the screen
    :type focal_distance: float or int
    :param axis: the axis which will be used for perspective
    :type axis: any type accepted by :func:`parse_3d_axis`
    :return: the projected points
    :rtype: numpy.ndarray *shape=(N, 3)*
    """
    p = np.array(points)  # Copy
    axis = parse_3d_axis(axis)
    factor = focal_distance / points[:, axis]
    p[:, axis].fill(focal_distance)

    for i in range(3):
        if i != axis:
            p[:, i] *= factor

    return p


def any_point_above(points, a, axis=2):
    """
    Returns true if a point in the array of points has a coordinate along given axis higher or equal to given threshold.

    :param points: the points
    :type points: numpy.ndarray *shape=(N, 3)*
    :param a: lower bound condition
    :type a: float
    :param axis: the axis which will be used for perspective
    :type axis: any type accepted by :func:`parse_3d_axis`
    :return: True if any point satisfies the condition
    :rtype: bool
    """
    points = points.reshape(-1, 3)
    axis = parse_3d_axis(axis)
    return np.any(a <= points[:, axis])


def any_point_below(points, b, axis=2):
    """
    Returns true if a point in the array of points has a coordinate along given axis lower or equal to given threshold.

    :param points: the points
    :type points: numpy.ndarray *shape=(N, 3)*
    :param b: upper bound condition
    :type b: float
    :param axis: the axis which will be used for perspective
    :type axis: any type accepted by :func:`parse_3d_axis`
    :return: True if any point satisfies the condition
    :rtype: bool
    """
    points = points.reshape(-1, 3)
    axis = parse_3d_axis(axis)
    return np.any(points[:, axis] <= b)


def any_point_between(points, a, b, axis=2):
    """
    Returns true if a point in the array of points has a coordinate along given axis between the given thresholds.

    :param points: the points
    :type points: numpy.ndarray *shape=(N, 3)*
    :param a: lower bound condition
    :type a: float
    :param b: upper bound condition
    :type b: float
    :param axis: the axis which will be used for perspective
    :type axis: any type accepted by :func:`parse_3d_axis`
    :return: True if any point satisfies the condition
    :rtype: bool
    """
    points = points.reshape(-1, 3)
    axis = parse_3d_axis(axis)
    return np.any(a <= points[:, axis] <= b)


class OrientedGeometry:
    """
    Disclaimer: this class and its subclasses are clearly inspired from the pakcage `pyny3d`. The main problem with this
    package is the lack of modularity and ability to subclass objects with ease; this is why no inheritance from this
    package is done, but a complete re-writing of useful functions has been preferred.

    An oriented geometry is any 3D geometry where each embedded is oriented.
    The orientation of the geometry is defined as this:

        "If your geometry has an inside and an outside, then the normal of each polygon contained in this geometry
        should be pointing outward. Watching the polygon from the outside, the loop passing through its points should be
        ccw (counter-clock-wise)."

    Oriented geometry constructors assume that their inputs are already oriented.
    """
    def __init__(self):
        self.domain = None
        self.centroid = None

    def get_polygons_iter(self):
        """
        Returns all the polygons contained in the geometry.

        :return: a generator of polygons
        :rtype: Generator[OrientedPolygon]
        """
        raise NotImplementedError

    def get_polygons_list(self):
        """
        Return all the polygons contained in the geometry.

        :return: a list of polygons
        :rtype: List[OrientedPolygon]
        """
        return list(self.get_polygons_iter())

    def get_polygons_count(self):
        """
        Returns the count of polygons in the geometry.

        :return: the count of polygons
        :rtype: int
        """
        return sum(1 for _ in self.get_polygons_iter())

    def get_polygons_matching(self, func, *args, func_args=(), **kwargs):
        """
        Returns all the polygons that match a given function.
        The function must take as first argument a polygon.

        :param func: the match function
        :type func: function
        :param args: extra objects to be filtered along with polygons
        :type args: Iterable[Any]
        :param func_args: positional arguments passed to `func`
        :param kwargs: keyword arguments passed to `func`
        :return: a generator of polygons
        :rtype: Generator[OrientedPolygon]
        """
        if len(args) != 0:
            for iterable in zip(self.get_polygons_iter(), *args):
                polygon = iterable[0]
                if func(polygon, *func_args, **kwargs):
                    yield iterable
        else:
            for polygon in self.get_polygons_iter():
                if func(polygon, *func_args, **kwargs):
                    yield polygon

    def apply_on_points(self, func, *args, **kwargs):
        """
        Applies a function recursively on all the geometries contained in this one.
        The function must take as first argument the points, as an numpy.ndarray with (N, 3) shape.

        :param func: the function
        :type func: function
        :param args: positional arguments passed to `func`
        :param kwargs: keyword argument passed to `func`
        :return: the new geometry
        :rtype OrientedGeometry
        """
        raise NotImplementedError

    def translate(self, vector):
        """
        Translates geometry using a vector as displacement.

        :param points: the points to translate
        :type points: numpy.ndarray *shape=(N, 3)*
        :param vector: the displacement vector
        :type vector: numpy.ndarray *size=3*
        :return: the new geometry
        :rtype OrientedGeometry
        """
        return self.apply_on_points(translate_points, vector)

    def project(self, matrix, around_point=None):
        """
        Projects a geometry on different axes given by matrix columns.

        :param matrix: the matrix to project all the geometry
        :type matrix: numpy.ndarray *shape=(3, 3)*
        :param around_point: if present, will apply the project around this point
        :type around_point: numpy.ndarray *size=3*
        :return: the new geometry
        :rtype OrientedGeometry
        """
        return self.apply_on_points(project_points, matrix, around_point=around_point)

    def project_with_perspective_mapping(self, focal_distance=1, axis=2):
        """
        Projects points with a perspective using similar triangles rule.
        The screen on which the points are projected is at a given distance from the origin.

        :param focal_distance: the distance to the screen
        :type focal_distance: float or int
        :param axis: the axis which will be used for perspective
        :type axis: any type accepted by :func:`parse_3d_axis`
        :return: the new geometry
        :rtype OrientedGeometry
        """
        return self.apply_on_points(project_points_with_perspective_mapping, focal_distance=focal_distance, axis=axis)

    def center_3d_plot(self, ax):
        """
        Centers and keep the aspect ratio in a 3D representation.

        :param ax: axes to apply the method.
        :type ax: mplot3d.Axes3D
        """

        domain = self.get_domain()
        bound = np.max(domain[1] - domain[0])
        centroid = self.get_centroid()
        pos = np.vstack((centroid - bound / 2, centroid + bound / 2))

        ax.set_xlim3d(left=pos[0, 0], right=pos[1, 0])
        ax.set_ylim3d(bottom=pos[0, 1], top=pos[1, 1])
        ax.set_zlim3d(bottom=pos[0, 2], top=pos[1, 2])

    def tight_3d_plot(self, ax):
        """
        Sets the axes limits to be tight on the geometry.

        :param ax: axes to apply the method.
        :type ax: mplot3d.Axes3D
        """
        domain = self.get_domain()
        ax.set_xlim([domain[0, 0], domain[1, 0]])
        ax.set_ylim([domain[0, 1], domain[1, 1]])
        ax.set_zlim([domain[0, 2], domain[1, 2]])

    def get_points(self):
        """
        Returns all the points in the geometry in a concatenated array.

        :return: the points
        :rtype: numpy.ndarray *shape=(N, 3)*
        """
        return np.concatenate([polygon.points for polygon in self.get_polygons_iter()])

    def get_domain(self):
        """
        Returns coordinates of the smallest prism containing this geometry.

        :return: opposite vertices of the bounding prism for this object.
        :rtype: numpy.ndarray([min], [max])
        """
        if self.domain is None:
            points = self.get_points()
            self.domain = np.array([points.min(axis=0),
                                    points.max(axis=0)])
        return self.domain

    def get_centroid(self):
        """
        The centroid is considered the center point of the circumscribed
        parallelepiped, not the mass center.

        :returns: (x, y, z) coordinates of the centroid of the object.
        :rtype: numpy.ndarray
        """
        if self.centroid is None:
            self.centroid = self.get_domain().mean(axis=0)

        return self.centroid


class OrientedPolygon(OrientedGeometry):
    """
    An oriented polygon is the simplest oriented geometry: it consists in an array of points, connected between their
    direct neighbours.

    :param points: the points
    :type points: numpy.ndarray *shape=(N, 3)*
    """
    def __init__(self, points):
        super().__init__()
        self.points = points

        self.parametric = None
        self.matrix = None

    def get_polygons_iter(self):
        yield self

    def distance_to_point(self, point):
        """
        Returns the distance from this polygon the a given point.

        :param point: the point
        :type point: ndarray *size=3*
        :return: the distance
        :rtype: float
        """
        # https://mathinsight.org/distance_point_plane
        normal = self.get_normal()
        v = point - self.points[0, :]
        return np.dot(normal, v.reshape(3))

    def apply_on_points(self, func, *args, **kwargs):
        return OrientedPolygon(func(self.points, *args, **kwargs))

    def magnify(self, point, factor=None, distance=None):
        """
        Returns an new polygon which is obtained by magnifying its size:
            *   if factor is present, will multiply the distance of all the points to the given point in the polygon by
                this factor
            *   else if distance is present, will set the distance between this polygon and the given point

        :param point: the reference point
        :type point: ndarray *size=3*
        :param factor: the multiplication factor
        :param distance: the required distance between the point and the polygon
        :type distance: float or int
        :return: the new polygon
        :rtype: OrientedPolygon
        """
        if factor is not None:
            pass
        elif distance is not None:
            d = self.distance_to_point(point)
            factor = - distance / d

        else:
            raise ValueError('factor and distance parameters can\'t de None at the same time.')

        point = point.reshape(1, 3)
        vectors = self.points - point
        vectors *= factor
        points = vectors + point
        return OrientedPolygon(points)

    def get_parametric(self):
        """
        Returns the parametric equation of the plane described by the polygon.
        It will return all four coefficients such that: a*x + b*y + c*z + d = 0.

        :return: the coefficients of the parametric equation
        :rtype: np.ndarray *shape=(4)*
        """
        if self.parametric is None:

            # Plane calculation
            normal = np.cross(self.points[1, :] - self.points[0, :],
                              self.points[2, :] - self.points[1, :])
            normal /= np.linalg.norm(normal)  # Normalize
            a, b, c = normal
            d = -np.dot(np.array([a, b, c]), self.points[2, :])
            self.parametric = np.array([a, b, c, d])
        return self.parametric

    def get_normal(self):
        """
        Returns the normal, as a unit vector, pointing outward of the polygon.

        :return: the normal vector
        :rtype: np.ndarray *shape=(3)
        """
        return self.get_parametric()[:-1]

    def get_matrix(self):
        """
        Returns a 3-by-3 orthogonal matrix where is column correspond to an axis of the polygon.
        matrix = [x, y, z] where
            x belongs to the polygon
            y belongs to the polygon
            z is normal to the polygon

        The axes follow the right hand side rule and will be normalized.

        In order to project points into the polygon's coordinates, use the transposed matrix !

        :return: the matrix of axes
        :rtype: ndarray *shape=(3, 3)*
        """
        if self.matrix is not None:
            return self.matrix

        points = self.points
        A = points[0, :]
        B = points[1, :]
        normal = self.get_normal()
        matrix = np.empty((3, 3), dtype=float)
        matrix[0, :] = B - A
        matrix[1, :] = np.cross(normal, matrix[0, :])
        matrix[2, :] = normal  # Already normalized

        for i in range(2):
            matrix[i, :] /= np.linalg.norm([matrix[i, :]])

        self.matrix = matrix.T

        return self.matrix

    def plot2d(self, *args, facecolor='b', alpha=1, edgecolor='k', lw=1, ret=False, ax=None, **kwargs):

        ax = plot_utils.get_2d_plot_ax(ax)

        plot_utils.add_polygon_to_2d_ax(ax, self.points, *args, facecolor=facecolor, alpha=alpha,
                                        edgecolor=edgecolor, lw=lw, **kwargs)

        if ret:
            return ax

    def plot3d(self, facecolor=(0, 0, 0, 0), edgecolor='k', alpha=0.1, ret=False, ax=None,
             normal=False, normal_kwargs=None,
             orientation=False, orientation_kwargs=None):

        if normal and normal_kwargs is None:
            normal_kwargs = {'color': 'b', 'length': 10}
        if orientation and orientation_kwargs is None:
            orientation_kwargs = {'color': 'r', 'arrow_length_ratio': 0.1}

        ax = plot_utils.get_3d_plot_ax(ax=ax)

        plot_utils.add_polygon_to_3d_ax(ax, self.points, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)

        center = self.get_centroid()

        if normal:
            x, y, z = center
            u, v, w = self.get_normal()
            ax.quiver(x, y, z, u, v, w, **normal_kwargs)

        if orientation:
            factor = 0.9
            n = self.points.shape[0]
            for i in range(n):
                A = self.points[i - 1, :]
                B = self.points[i, :]
                CA = factor * (A - center)
                CB = factor * (B - center)
                u, v, w = CB - CA
                x, y, z = center + CA
                ax.quiver(x, y, z, u, v, w, **orientation_kwargs)

        if ret:
            return ax


class OrientedSurface(OrientedGeometry):
    """
    An oriented surface is a combination of oriented polygons.

    :param polygons: the polygons
    :type polygons: a list of (or an instance of) numpy.ndarray (points) or OrientedPolygons
    """
    def __init__(self, polygons):
        super().__init__()

        if type(polygons) != list:
            polygons = [polygons]

        if type(polygons[0]) == np.ndarray:
            self.polygons = [OrientedPolygon(polygon)
                             for polygon in polygons]
        elif isinstance(polygons[0], OrientedPolygon):
            self.polygons = polygons
        else:
            raise ValueError('OrientedSurface needs a nump.ndarray or OrientedPolygon as input')

    def get_polygons_iter(self):
        return iter(self.polygons)

    def apply_on_points(self, func, *args, **kwargs):
        projected_polygons = [polygon.apply_on_points(func, *args, **kwargs)
                              for polygon in self.polygons]
        return OrientedSurface(projected_polygons)

    def plot2d(self, *args, ret=False, ax=None, **kwargs):

        ax = plot_utils.get_2d_plot_ax(ax)

        for polygon in self.polygons:
            ax = polygon.plot2d(*args, ax=ax, ret=True, **kwargs)

        if ret:
            return ax

    def plot3d(self, *args, ret=False, ax=None, **kwargs):

        ax = plot_utils.get_3d_plot_ax(ax)

        for polygon in self.polygons:
            polygon.plot3d(*args, ax=ax, ret=False, **kwargs)

        if ret:
            return ax


class OrientedPolyhedron(OrientedGeometry):
    """
    An oriented polyhedron is a polyhedron with an "inside" and an "outside".
    It is composed with oriented polygons.
    Surfaces are oriented ccw where watched from the outside. Normal vectors are pointing outward.
    """
    def __init__(self, polygons, **kwargs):
        super().__init__()
        self.aux_surface = OrientedSurface(polygons, **kwargs)
        self.polygons = self.aux_surface.polygons

    def get_polygons_iter(self):
        return iter(self.polygons)

    def apply_on_points(self, func, *args, **kwargs):
        surface = self.aux_surface.apply_on_points(func, *args, **kwargs)
        return OrientedPolyhedron(surface.polygons)

    def plot2d(self, *args, ret=False, ax=None, **kwargs):
        ax = self.aux_surface.plot2d(*args, ret=True, ax=ax, **kwargs)

        if ret:
            return ax

    def plot3d(self, *args, ret=False, ax=None, **kwargs):
        ax = self.aux_surface.plot3d(*args, ret=True, ax=ax, **kwargs)

        if ret:
            return ax


class Pyramid(OrientedPolyhedron):
    """
    A pyramid is an oriented polyhedron described by a base polygon and an isolated point.
    """
    @staticmethod
    def by_point_and_polygon(point, polygon):
        """
        Returns a new pyramid from a base polygon and an isolated point.
        All the other polygons are constructed by joining points in the polygon to the isolated point.

        :param point: the isolated point
        :type point: numpy.ndarray *size=3*
        :param polygon: the base polygon
        :type polygon: OrientedPolygon
        :return: the new pyramid
        :rtype: Pyramid
        """

        base_points = polygon.points
        polygons = [polygon]

        n = base_points.shape[0]
        A = point

        for i in range(n):
            B = base_points[i - 1, :]
            C = base_points[i, :]
            polygon = OrientedPolygon(np.row_stack([C, B, A]))  # ccw
            polygons.append(polygon)

        return Pyramid(polygons)


class Building(OrientedPolyhedron):
    """
    A building is an oriented polyhedron constructed by extruding a polygon in the z direction.
    It consists in 2 flat faces, one for ground and one for rooftop, and as many other vertical faces are there
    are vertices in the original polygon.
    """

    @staticmethod
    def by_polygon_and_height(polygon, height, make_ccw=True, keep_ground=True):
        """
        Constructs a building from a 3D polygon on the ground.

        :param polygon: 3D polygon
        :type polygon: OrientedPolygon or numpy.ndarray *shape=(N, 3)*
        :param height: the height of the building
        :type height: float or int
        :param make_ccw: if True, ensure that polygon is oriented correctly
        :type make_ccw: bool
        :param keep_ground: if True, will keep the ground polygon in the building
        :type keep_ground: bool
        :return: a building
        :rtype: Building
        """

        if isinstance(polygon, OrientedPolygon):
            bottom_points = polygon.points
        elif isinstance(polygon, np.ndarray):
            bottom_points = polygon
        else:
            raise ValueError(f'Type {type(polygon)} is not supported for polygon !')

        top_points = np.array(bottom_points)
        top_points[:, -1] += float(height)

        if make_ccw:
            if not is_ccw(top_points):
                top_points = top_points[::-1, :]
            else:
                bottom_points = bottom_points[::-1, :]

        top = OrientedPolygon(top_points)
        bottom = OrientedPolygon(bottom_points)

        if top.get_normal()[2] < 0:  # z component should be positive
            top.parametric = - top.parametric
        if bottom.get_normal()[2] > 0:  # z component should be negative
            bottom.parametric = - bottom.parametric

        n = top_points.shape[0]

        if keep_ground:
            polygons = [top, bottom]
        else:
            polygons = [top]

        bottom_points = bottom_points[::-1, :]  # Bottom points are now oriented cw to match top points

        # For each face other than top and bottom
        for i in range(n):
            A = top_points[i - 1, :]
            B = top_points[i, :]
            C = bottom_points[i - 1, :]
            D = bottom_points[i, :]

            face_points = np.row_stack([A, C, D, B])

            polygon = OrientedPolygon(face_points)
            polygons.append(polygon)

        return Building(polygons)

    @staticmethod
    def by_polygon2d_and_height(polygon, height, make_ccw=True, keep_ground=False):
        """
        Constructs a building from a 2D polygon.

        :param polygon: 2D polygon
        :type polygon: Polygon (shapely), OrientedPolygon or numpy.ndarray *shape=(N, 3)*
        :param height: the height of the building
        :type height: float or int
        :param make_ccw: if True, ensure that polygon is oriented correctly
        :type make_ccw: bool
        :param keep_ground: if True, will keep the ground polygon in the building
        :type keep_ground: bool
        :return: a building
        :rtype: Building
        """

        if isinstance(polygon, OrientedPolygon):
            x, y = polygon.points[:, :-1]
        elif isinstance(polygon, shPolygon):
            x, y = polygon.exterior.coords.xy
            x = x[:-1]
            y = y[:-1]
        elif isinstance(polygon, np.ndarray):
            x, y = polygon[:, :-1]
        else:
            raise ValueError(f'Type {type(polygon)} is not supported for polygon !')

        z0 = np.zeros_like(x, dtype=float)
        bottom_points = np.column_stack([x, y, z0])

        polygon = OrientedPolygon(bottom_points)

        return Building.by_polygon_and_height(polygon, height, make_ccw=make_ccw, keep_ground=keep_ground)


class Cube(OrientedPolyhedron):
    """
    A cube is an oriented polyhedron that can be fully described by its center and one of its 6 faces.
    """

    @staticmethod
    def by_point_and_side_length(point, side_length):
        """
        Creates a cube from an origin point and a side length.

        :param point: the center of the cube
        :type point: ndarray *size=3*
        :param side_length: the length of one side of the cube
        :type side_length: float
        :return: a cube
        :rtype: Cube
        """
        r = side_length / 2
        polygon = np.array([
            [r, r, -r],
            [-r, r, -r],
            [-r, -r, -r],
            [r, -r, -r]
        ])

        polygon += point.reshape(1, 3)

        building = Building.by_polygon_and_height(polygon, side_length)

        return Cube(building.polygons)


class OrientedPlace(OrientedGeometry):

    def __init__(self, surface, polyhedra=[], set_of_points=np.empty((0, 3))):
        super().__init__()

        if isinstance(surface, OrientedSurface):
            self.surface = surface
        elif type(surface) == list or type(surface) == np.ndarray:
            self.surface = OrientedSurface(surface)
        else:
            raise ValueError('OrientedPlace needs an OrientedSurface or a numpy.ndarray as surface input')

        if polyhedra != []:
            if type(polyhedra) != list:
                polyhedra = [polyhedra]
            if isinstance(polyhedra[0], OrientedPolyhedron):
                self.polyhedra = polyhedra
            else:
                self.polyhedra = [
                    OrientedPolyhedron(polyhedron) for polyhedron in polyhedra
                ]
        else:
            self.polyhedra = []

        if type(set_of_points) == np.ndarray:
            if set_of_points.shape[1] == 3:
                self.set_of_points = set_of_points
        else:
            raise ValueError('OrientedPlace has an invalid set_of_points as input')

    def add_set_of_points(self, points):
        """
        Adds a set of points to the current set of points in the place.

        :param points: the points to add
        :type points: numpy.ndarray *shape=(N, 3)*
        """
        self.set_of_points = np.concatenate([self.set_of_points, points])

    def get_polygons_iter(self):
        return itertools.chain(
            self.surface.get_polygons_iter(),
            itertools.chain.from_iterable(
                polyhedron.get_polygons_iter() for polyhedron in self.polyhedra
            )
        )

    def apply_on_points(self, func, *args, **kwargs):
        surface = self.surface.apply_on_points(func, *args, **kwargs)
        polyhedra = [polyhedron.apply_on_points(func, *args, **kwargs)
                     for polyhedron in self.polyhedra]
        set_of_points = func(self.set_of_points, *args, **kwargs)
        return OrientedPlace(surface, polyhedra=polyhedra, set_of_points=set_of_points)

    def plot2d(self, ret=False, ax=None,
               poly_args=None, poly_kwargs=None,
               points_args=None, points_kwargs=None):

        if poly_args is None:
            poly_args = ()
        if poly_kwargs is None:
            poly_kwargs = {}
        if points_args is None:
            points_args = ()
        if points_kwargs is None:
            points_kwargs = {}

        ax = plot_utils.get_2d_plot_ax(ax)

        self.surface.plot2d(*poly_args, ax=ax, ret=False, **poly_kwargs)

        for polyhedron in self.polyhedra:
            polyhedron.plot2d(*poly_args, ax=ax, ret=False, **poly_kwargs)

        if self.set_of_points.size > 0:
            points = self.set_of_points
            ax.scatter(points[:, 0], points[:, 1], *points_args, **points_kwargs)

        if ret:
            return ax

    def plot3d(self, ret=False, ax=None,
               poly_args=None, poly_kwargs=None,
               points_args=None, points_kwargs=None):

        if poly_args is None:
            poly_args = ()
        if poly_kwargs is None:
            poly_kwargs = {}
        if points_args is None:
            points_args = ()
        if points_kwargs is None:
            points_kwargs = {}

        ax = plot_utils.get_3d_plot_ax(ax)

        self.surface.plot3d(*poly_args, ax=ax, ret=False, **poly_kwargs)

        for polyhedron in self.polyhedra:
            polyhedron.plot3d(*poly_args, ax=ax, ret=False, **poly_kwargs)

        if self.set_of_points.size > 0:
            points = self.set_of_points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], *points_args, **points_kwargs)

        if ret:
            return ax


def generate_place_from_rooftops_file(roof_top_file, center=True):
    """
    Returns a place from a geographic file containing building rooftops and their height.

    :param roof_top_file: file containing polygons describing buildings on ground, with height attribute
    :type roof_top_file: str, any filetype supported by :func:`geopandas.read_file`
    :param center: if True, will center the coordinates
    :type center: bool
    :return: a place containing the buildings and a flat ground surface covering the whole area
    :rtype: OrientedPlace
    """
    gdf = gpd.read_file(roof_top_file)
    gdf.dropna(subset=['height'], inplace=True)
    gdf.to_crs(epsg=3035, inplace=True)  # To make buildings look more realistic, there may be a better choice :)

    if center:
        bounds = gdf.total_bounds
        x = (bounds[0] + bounds[2]) / 2
        y = (bounds[1] + bounds[3]) / 2

        gdf['geometry'] = gdf['geometry'].translate(-x, -y)

    def func(series: gpd.GeoSeries):
        return Building.by_polygon2d_and_height(series['geometry'], series['height'], keep_ground=False)

    polyhedra = gdf.apply(func, axis=1).values.tolist()

    bounds = gdf.total_bounds.reshape(2, 2)

    points = np.zeros((4, 3))
    points[0::3, 0] = bounds[0, 0]
    points[1:3, 0] = bounds[1, 0]
    points[:2, 1] = bounds[0, 1]
    points[2:, 1] = bounds[1, 1]

    ground_surface = OrientedSurface(points)

    place = OrientedPlace(ground_surface)
    place.polyhedra = polyhedra

    return place


if __name__ == '__main__':

    # 1. Load data

    place = generate_place_from_rooftops_file('../data/small.geojson')

    # 2. Create TX and RX

    domain = place.get_domain()
    ground_center = place.get_centroid()

    tx = ground_center + [-50, 5, 1]
    rx = ground_center + [-2, 0, 5]
    tx = tx.reshape(1, 3)
    rx = rx.reshape(1, 3)

    # 2.1 Create a cube around TX

    cube = Cube.by_point_and_side_length(tx, 10)

    # 2.1.1 Rotate this cube around its center
    from scipy.spatial.transform import Rotation as R

    rot2 = R.from_euler('xyz', [0, 10, 10], degrees=True).as_matrix()

    cube = cube.project(rot2, around_point=tx)

    # 2.2 Place TX and RX in the 'place'
    #place.add_set_of_points(tx)
    place.add_set_of_points(rx)

    # 2.3 Translate the geometry around TX

    place = place.translate(-tx)
    cube = cube.translate(-tx)
    polygons = place.get_polygons_list()

    # 3. Plot the whole geometry
    ax1 = place.plot3d(ret=True)
    cube.plot3d(ax=ax1)

    place.center_3d_plot(ax1)

    # 3.1 Picking one face of the cube as the screen and coloring it
    screen = cube.polygons[2]

    screen.plot3d(facecolor='g', alpha=0.5, ax=ax1, orientation=True, normal=True)
    # 4. Create the screen on which geometry will be projected
    distance = 5  # Distance from TX to screen

    #screen = screen.magnify(tx * 0, distance=distance)  # TX is now at [0, 0, 0]

    print('Screen points:\n', screen.points)

    # 5. First, changing geometry coordinates to match screen's orientation
    matrix = screen.get_matrix().T

    print('Coordinates matrix:\n', matrix)
    place = place.project(matrix)
    screen = screen.project(matrix)
    print('Screen points after transformation:\n', screen.points)

    ax = place.plot3d(ret=True)
    cube.plot3d(ax=ax)
    screen.plot3d(facecolor='g', ax=ax)


    def filter_func(polygon):
        return np.dot(polygon.get_normal(), screen.get_normal()) < np.arccos(np.pi/4) and any_point_above(polygon.points, 0, axis=2)

    poly = place.get_polygons_matching(filter_func, polygons)

    for _, polygon in poly:
        polygon.plot3d(ax=ax1, edgecolor='r')

    #print(len(list(poly)))
    # 6. Perspective mapping on z direction
    place = place.project_with_perspective_mapping(focal_distance=distance)

    screen = screen.project_with_perspective_mapping(focal_distance=distance)
    print('Screen points:\n', screen.points)

    rot = R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()

    ax = place.project(rot).plot2d(ret=True, poly_kwargs=dict(alpha=0.5))

    screen.project(rot).plot2d(ax=ax, facecolor='g', alpha=0.4)
    print(len(list(place.get_polygons_iter())))
    plt.axis('equal')

    plt.show()






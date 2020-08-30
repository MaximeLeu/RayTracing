# Plotting libraries
from radarcoverage import plot_utils

# Numerical libraries
import numpy as np
from numpy.dual import norm
from scipy.optimize import root

# Geometry libraries
from shapely.geometry import Polygon as shPolygon
from shapely.geometry import LineString as shLine
from shapely.geometry import Point as shPoint
import geopandas as gpd

# Utils
import itertools
import pickle


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
    :rtype: int
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


def reflexion_on_plane(incidents, normal, normalized=False):
    """
    Return the reflexion of incident vector on a plane with given normal.
    See details: https://en.wikipedia.org/wiki/Reflection_(mathematics)

    :param incidents: incident vectors
    :type incidents: numpy.ndarray *shape=(N, 3)*
    :param normal: normal vector to the plane
    :type normal: numpy.ndarray *size=3*
    :param normalized: if True, assume normal vector is a unit vector (accelerates computation)
    :type normalized: bool
    :return: the reflected vector(s)
    :rtype: numpy.ndarray *shape(N, 3)*
    """
    normal = normal.reshape((1, 3))
    incidents = incidents.reshape((-1, 3))
    if normalized:
        den = 1
    else:
        den = normal @ normal.T
    return incidents - (incidents @ normal.T) @ ((2 / den) * normal)  # Order of operation minimizes the # of op.


def reflexion_points_from_origin_destination_and_planes(origin, destination, planes_parametric, **kwargs):
    """
    Returns the reflection point on each plane such that a path between origin and destination is possible.
    The parametric equation of the plane should contained the normal vector in a normalized form.

    :param origin: the origin point
    :type origin: numpy.ndarray *size=3*
    :param destination: the destination point
    :type destination: numpy.ndarray *size=3*
    :param planes_parametric: the coefficients of the parametric equation of each plane
    :type planes_parametric: list of numpy.ndarray *shape=(4)*
    :param kwargs: keyword parameters passed to :func:`scipy.optimize.fsolve`
    :type kwargs: any
    :return: the reflection point on each of the planes and the solution
    :rtype: numpy.array *shape(1, 3)*, scipy.optimize.OptimizeResult
    """
    A = origin.reshape(1, 3)
    B = destination.reshape(1, 3)
    parametric = np.vstack(planes_parametric)
    normal = parametric[:, :3]
    d = parametric[:, 3].reshape(-1, 1)

    n = parametric.shape[0]

    points = np.empty((n + 2, 3), dtype=float)
    points[0, :] = A
    points[-1, :] = B

    for i in range(1, n + 1):
        # First guess for solution
        points[i, :] = 0.5 * (points[i - 1, :] + B - 2 * (d[i-1] + points[i - 1, :] @ normal[i-1, :].T) * normal[i-1, :])

    def gamma(v):
        norms = norm(v, axis=1)
        return norms[:-1] / norms[1:]

    def func(x):
        points[1:n+1, :] = x.reshape(-1, 3)
        v = np.diff(points, axis=0)
        g = gamma(v).reshape(-1, 1)

        dot_product = np.einsum('ij,ij->i', points[:n, :], normal).reshape(-1, 1)

        return (g * v[1:, :] - v[:-1, :] - 2 * (d + dot_product) * normal).reshape(-1)

    sol = root(func, x0=points[1:n + 1, :].reshape(-1), **kwargs)

    x = sol.x.reshape(-1, 3)

    return x, sol


def diffraction_point_from_origin_destination_and_edge(origin, destination, edge, **kwargs):
    """
    Returns the diffraction point on a edge such that a path between origin and destination is possible.

    :param origin: the origin point
    :type origin: numpy.ndarray *size=3*
    :param destination: the destination point
    :type destination: numpy.ndarray *size=3*
    :param edge: the edge in which reflexion is done
    :type edge: numpy.ndarray *shape=(2, 3)*
    :param kwargs: keyword parameters passed to :func:`scipy.optimize.root`
    :type kwargs: any
    :return: the diffraction point on the edge and the solution
    :rtype: numpy.array *shape(1, 3)*, scipy.optimize.OptimizeResult
    """
    A = origin.reshape(1, 3)
    B = destination.reshape(1, 3)

    matrix = projection_matrix_from_line_path(edge).T

    projected_A = project_points(A, matrix)[0, :]
    projected_B = project_points(B, matrix)[0, :]
    projected_edge = project_points(edge, matrix)
    Xe = projected_edge[0, :]
    ze = Xe[2]

    def func(x):
        t = x - ze
        x = np.copy(Xe)
        x[2] += t
        i = x - projected_A
        d = projected_B - x

        num1 = (ze - projected_A[2] + t)
        num2 = (projected_B[2] - ze - t)
        den1 = norm(i)
        den2 = norm(d)

        f = num1 / den1 - num2 / den2

        den1_prime = num1 / den1
        den2_prime = - num2 / den2

        jac = (-num1 - den1_prime) / (den1 * den1) - (num2 - den2_prime) / (den2 * den2)

        return f, jac

    sol = root(func, x0=ze, jac=True, **kwargs)
    Xe[2] = sol.x

    return project_points(Xe, matrix.T), sol


def reflexion_points_and_diffraction_point_from_origin_destination_planes_and_edge(origin, destination,
                                                                                   planes_parametric, edge, **kwargs):
    if len(planes_parametric) == 0:
        return diffraction_point_from_origin_destination_and_edge(origin, destination, edge, **kwargs)
    elif edge is None:
        return reflexion_points_from_origin_destination_and_planes(origin, destination, planes_parametric, **kwargs)

    A = origin.reshape(1, 3)
    B = destination.reshape(1, 3)

    # Reflection
    parametric = np.vstack(planes_parametric)
    normal = parametric[:, :3]
    d = parametric[:, 3].reshape(-1, 1)
    n = parametric.shape[0]

    # Diffraction
    matrix = projection_matrix_from_line_path(edge).T

    projected_B = project_points(B, matrix)[0, :]
    projected_edge = project_points(edge, matrix)
    Xe = projected_edge[0, :]
    ze = Xe[2]

    points = np.empty((n + 2, 3), dtype=float)
    points[0, :] = A
    points[-1, :] = edge[0, :]

    for i in range(1, n + 1):
        # First guess for solution
        points[i, :] = 0.5 * (
                    points[i - 1, :] + B - 2 * (d[i - 1] + points[i - 1, :] @ normal[i - 1, :].T) * normal[i - 1, :])

    def gamma(v):
        norms = norm(v, axis=1)
        gamma = norms[:-1] / norms[1:]
        return gamma

    def func(x):
        r = np.empty_like(x)
        # Reflection
        points[1:n+1, :] = x[:-1].reshape(-1, 3)
        points[-1, :] = project_points(np.array([Xe[0], Xe[1], x[-1]]), matrix.T)

        v = np.diff(points, axis=0)
        g = gamma(v).reshape(-1, 1)

        dot_product = np.einsum('ij,ij->i', points[:n, :], normal).reshape(-1, 1)

        projected_A = project_points(points[-2, :], matrix)

        r[:-1] = (g * v[1:, :] - v[:-1, :] - 2 * (d + dot_product) * normal).reshape(-1)

        # Diffraction
        t = x[-1] - ze
        xn = np.copy(Xe)
        xn[2] += t
        i_ = xn - projected_A
        d_ = projected_B - xn

        num1 = (ze - projected_A[2] + t)
        num2 = (projected_B[2] - ze - t)
        den1 = norm(i_)
        den2 = norm(d_)

        r[-1] = num1 / den1 - num2 / den2

        return r

    sol = root(func, x0=points[1:, :].flat[:-2], **kwargs)

    x = sol.x

    points[1:n+1, :] = x[:-1].reshape(-1, 3)
    points[-1, :] = project_points(np.array([Xe[0], Xe[1], x[-1]]), matrix.T)

    return points[1:, :], sol


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


def project_points_on_spherical_coordinates(points, r_axis=2):
    r_axis = parse_3d_axis(r_axis)
    p = np.array(points, ndmin=2)  # Copy
    spherical_points = np.empty(p.shape, dtype=float)
    spherical_points[:, 0] = norm(p, axis=1)
    spherical_points[:, 1] = np.arctan2(p[:, 1], p[:, 0])
    spherical_points[:, 2] = np.arccos(p[:, 2] / spherical_points[:, 0])

    return np.roll(spherical_points, r_axis, axis=1)


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
    p = np.array(points, ndmin=2)  # Copy
    axis = parse_3d_axis(axis)
    factor = focal_distance / p[:, axis]

    for i in range(3):
        if i != axis:
            p[:, i] *= factor

    return p


def restore_points_before_projective_mapping(points, focal_distance=1, axis=2):
    """
    Reverses a projective mapping for points. Only possible if data along given axis has been kept.

    :param points: the points to be restored
    :type points: numpy.ndarray *shape=(N, 3)*
    :param focal_distance: the distance to the screen
    :type focal_distance: float or int
    :param axis: the axis which will be used for perspective
    :type axis: any type accepted by :func:`parse_3d_axis`
    :return: the restored points
    :rtype: numpy.ndarray *shape=(N, 3)*
    """
    p = np.array(points)  # Copy
    axis = parse_3d_axis(axis)
    factor = points[:, axis] / focal_distance

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


def projection_matrix_from_line_path(points):
    """
    Returns an orthogonal matrix which can be used to project any set of points in a coordinates system aligned with
    given line. The last axis is the axis with the same direction as the line.

    :param points: two points describing line path
    :type points: numpy.ndarray *shape=(2, 3)*
    :return: a matrix
    :rtype: numpy.ndarray *shape=(3, 3)*
    """
    points = points.reshape(-1, 3)
    A = points[0, :]
    B = points[1, :]
    V = B - A
    z = V / norm(V)
    y = np.array([0, 1, 0], dtype=float)  # Arbitrary
    y -= y.dot(z) * z
    y /= norm(y)
    x = np.cross(y, z)
    return np.row_stack([x, y, z]).T


def polygons_obstruct_line_path(polygons, points):
    """
    Returns whether a line path is obstructed by any of the polygons.

    :param polygons: the polygons:
    :type polygons: Iterable[OrientedPolygon]
    :param points: two points describing line path
    :type points: numpy.ndarray *shape=(2, 3)*
    :return: True if any polygon intercepts the line path
    :rtype: bool
    """
    matrix = projection_matrix_from_line_path(points).T

    projected_points = project_points(points, matrix)
    pA = projected_points[0, :]
    pB = projected_points[1, :]
    line = shLine([pA, pB])
    z_min = pA[2]
    z_max = pB[2]

    tol = 1e-8

    for polygon in polygons:
        projected_polygon = polygon.project(matrix)
        domain = projected_polygon.get_domain()

        if domain[1, 2] > z_min and domain[0, 2] < z_max:
            if projected_polygon.get_shapely().intersects(line):  # Projection of line is intersected by polygon
                normal = polygon.get_normal()
                d = polygon.get_parametric()[3]
                t = - (d + np.dot(A, normal)) / np.dot(V, normal)
                if tol < t < 1 - tol:  # Is the polygon between A and B ?
                    return True
    return False


def polygon_visibility_vector(polygon_A, polygons, out=None, strict=False):
    """
    Returns the visibility vector for a polygon facing N given polygons.
    For each element i, the vector tells if it is physically possible that a ray pointing outward the polygon_A's
    face could intercept an other polygon[i].

    As this problem is known to be NP complete, two approaches are offered:
        1. the `strict` approach makes no compromise and will only remove polygons that are 100% for sure not visible
        2. the other approach is to only take the polygons visible from the centroid of the polygon, therefore not
            taking into account all the possible paths and, thus, maybe removing visible polygons

    :param polygon_A: the reference polygon
    :type polygon_A: OrientedPolygon
    :param polygons: the polygons
    :type polygons: Iterable[OrientedPolygons]
    :param out: if provided, will store the result in this array
    :type out: None or numpy.ndarray *dtype=bool, shape=(N)*
    :param strict: if True, will choose strict approach
    :type strict: bool
    :return: the visibility vector
    :rtype: numpy.ndarray *dtype=bool, shape=(N)*
    """
    # Tolerances : the lower, the more polygons are considered visible, but some badly
    tol_dz = 1e-4  # Tolerance on z coordinate variation
    tol_dot = 1e-4  # Tolerance on dot product result
    tol_area = 1e-4

    # Output of data can be stored in a given array
    polygons = list(polygons)
    if out is None:
        visibility_vector = np.empty(len(polygons), dtype=bool)
    else:
        visibility_vector = out

    # Translating and projecting all polygons in polygon A coordinate system
    matrix_A = polygon_A.get_matrix().T
    centroid_A = polygon_A.get_centroid()

    projected_polygon_A = polygon_A.translate(-centroid_A).project(matrix_A)
    projected_polygons = [(i, polygon.translate(-centroid_A).project(matrix_A))
                          for i, polygon in enumerate(polygons) if polygon != polygon_A]

    # First filter: by z coordinate and vector analysis
    filtered_polygons = list()
    for i, projected_polygon_B in projected_polygons:
        if projected_polygon_B.get_domain()[1, 2] > tol_dz:
            centroid_B = polygons[i].get_centroid()

            # For each point in polygon A, we check if any can produce a rays bouncing on polygon B
            for point in polygon_A.points:
                vector = point - centroid_B
                vector /= norm(vector)
                if np.dot(vector, polygons[i].get_normal()) > tol_dot:
                    if strict:
                        visibility_vector[i] = True
                    filtered_polygons.append((i, projected_polygon_B.project_on_spherical_coordinates()))
                    break

    if strict:
        return visibility_vector

    def func(polygon):
        domain = polygon.get_domain()[0]
        return domain[2], norm(polygon.get_centroid())

    polygons_filtered = sorted(filtered_polygons, key=lambda x: func(x[1]))

    screen = polygon_A.project_on_spherical_coordinates().get_shapely()

    canvas = shPolygon()
    for i, projected_polygon_B in polygons_filtered:
        shapely_polygon_B = projected_polygon_B.project_with_perspective_mapping().get_shapely()
        if not shapely_polygon_B.is_valid:
            continue
        try:
            polygon_B_in_screen = shapely_polygon_B
        except:
            continue
        difference = polygon_B_in_screen.difference(canvas)
        if not difference.is_empty:
            canvas = canvas.union(polygon_B_in_screen)
            visibility_vector[i] = True

    return visibility_vector


def polygons_visibility_matrix(polygons, strict=False):
    """
    Returns the visibility matrix for N given polygons.
    For each row i, the matrix tells if it is physically possible that a ray pointing outward the polygon[i]'s
    face could intercept an other polygon[j]. See :func:`polygon_visibility_vector`'s documentation for more
    information.

    :param polygons: the polygons
    :type polygons: Iterable[OrientedPolygons]
    :param strict: if True, will choose strict approach
    :type strict: bool
    :return: the visibility matrix
    :rtype: numpy.ndarray *dtype=bool, shape=(N, N)*
    """
    polygons = list(polygons)
    n = len(polygons)

    visibility_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        polygon_A = polygons[i]
        polygon_visibility_vector(polygon_A, polygons, out=visibility_matrix[i, :], strict=strict)

    # Symmetric matrix
    #visibility_matrix = visibility_matrix | visibility_matrix.T

    return visibility_matrix


class OrientedGeometry(object):
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
    id = 0

    def __init__(self):
        self.id = OrientedGeometry.id
        OrientedGeometry.id += 1
        self.domain = None
        self.centroid = None
        self.visibility_matrix = None
        self.pause = False

    def __eq__(self, other):
        return self.id == other.id

    def save(self, filename):
        """
        Saves an oriented geometry object into a .ogeom file.

        :param filename: the filepath
        :type filename: str
        """
        if not filename.endswith('.ogeom'):
            filename += '.ogeom'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads an oriented geometry from a .ogeom file

        :param filename: the filepath
        :type filename: str
        :return: the geometry stored in the file
        :rtype: OrientedGeometry
        """
        if not filename.endswith('.ogeom'):
            raise ValueError(f'Can only read .ogeom files.')

        with open(filename, 'rb') as f:
            return pickle.load(f)

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
        :rtype: OrientedGeometry
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
        :rtype: OrientedGeometry
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
        :rtype: OrientedGeometry
        """
        return self.apply_on_points(project_points, matrix, around_point=around_point)

    def project_on_spherical_coordinates(self, r_axis=2):
        return self.apply_on_points(project_points_on_spherical_coordinates, r_axis=r_axis)

    def project_with_perspective_mapping(self, focal_distance=1, axis=2):
        """
        Projects points with a perspective using similar triangles rule.
        The screen on which the points are projected is at a given distance from the origin.

        :param focal_distance: the distance to the screen
        :type focal_distance: float or int
        :param axis: the axis which will be used for perspective
        :type axis: any type accepted by :func:`parse_3d_axis`
        :return: the new geometry
        :rtype: OrientedGeometry
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

    def get_domain(self, force=False):
        """
        Returns coordinates of the smallest prism containing this geometry.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: opposite vertices of the bounding prism for this object
        :rtype: numpy.ndarray([min], [max])
        """
        if force or self.domain is None:
            points = self.get_points()
            self.domain = np.array([points.min(axis=0),
                                    points.max(axis=0)])
        return self.domain

    def get_centroid(self, force=False):
        """
        The centroid is considered the center point of the circumscribed
        parallelepiped, not the mass center.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :returns: (x, y, z) coordinates of the centroid of the object
        :rtype: numpy.ndarray
        """
        if force or self.centroid is None:
            self.centroid = self.get_domain().mean(axis=0)

        return self.centroid

    def get_visibility_matrix(self, strict=False, force=False):
        """
        Returns the symmetric visibility matrix for the polygons in this geometry.
        See :func:`polygons_visibility_matrix`'s documentation for more information.


        :param strict: if True, will choose strict approach
        :type strict: bool
        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: the visibility matrix
        :rtype: numpy.ndarray *dtype=bool, shape=(N, N)*
        """
        if force or self.visibility_matrix is None:
            self.visibility_matrix = polygons_visibility_matrix(self.get_polygons_iter(), strict=strict)

        return self.visibility_matrix

    def show_visibility_matrix_animation(self, strict=False):
        """
        Shows a 3D animation of the visibility matrix by showing, for each polygon, the polygon face "observing" is blue
        and and the polygon faces that are visible to it in red.

        :param strict: if True, will choose strict approach
        :type strict: bool
        """
        ax = self.plot3d(ret=True)
        self.center_3d_plot(ax)
        polys3d = ax.collections
        visibility_matrix = self.get_visibility_matrix(strict=strict)

        n = len(polys3d)
        indices = itertools.cycle(itertools.chain.from_iterable(itertools.repeat(i, 30) for i in range(n)))

        def func(_):
            i = next(indices)

            for poly in polys3d:
                poly.set_alpha(0)

            for j in np.where(visibility_matrix[i])[0]:
                polys3d[j].set_facecolor('r')
                polys3d[j].set_alpha(0.5)

            polys3d[i].set_facecolor('b')
            polys3d[i].set_alpha(0.8)

        plot_utils.animate_3d_ax(ax, func=func)


class OrientedPolygon(OrientedGeometry):
    """
    An oriented polygon is the simplest oriented geometry: it consists in an array of points, connected between their
    direct neighbours.

    :param points: the points
    :type points: numpy.ndarray *shape=(N, 3)*
    """
    def __init__(self, points):
        super().__init__()
        self.points = points.astype(float)

        self.parametric = None
        self.matrix = None
        self.shapely = None

    def get_shapely(self, force=False):
        """
        Returns a 2D polygon from this polygon by removing the z component.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: the 2D polygon
        :rtype: shapely.geometry.Polygon
        """
        if force or self.shapely is None:
            self.shapely = shPolygon(self.points[:, :2])
            """
            if not self.shapely.is_valid:
                from scipy.spatial import ConvexHull
                from scipy.spatial.qhull import QhullError
                print('points', self.points)
                try:
                    hull = ConvexHull(self.points[:, :2])
                    points = self.points[hull.vertices, :2]
                    self.shapely = shPolygon(points)
                except QhullError:
                    pass
            """
        return self.shapely

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

    def get_parametric(self, force=False):
        """
        Returns the parametric equation of the plane described by the polygon.
        It will return all four coefficients such that: a*x + b*y + c*z + d = 0.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: the coefficients of the parametric equation
        :rtype: np.ndarray *shape=(4)*
        """
        if force or self.parametric is None:

            # Plane calculation
            normal = np.cross(self.points[1, :] - self.points[0, :],
                              self.points[2, :] - self.points[1, :])
            normal /= norm(normal)  # Normalize
            a, b, c = normal
            d = -np.dot(np.array([a, b, c]), self.points[2, :])
            self.parametric = np.array([a, b, c, d])
        return self.parametric

    def get_normal(self):
        """
        Returns the normal, as a unit vector, pointing outward of the polygon.

        :return: the normal vector
        :rtype: np.ndarray *shape=(3)*
        """
        return self.get_parametric()[:3]

    def contains_point(self, point, check_in_plane=False, plane_tol=1e-9):
        """
        Returns true if point belongs to polygon. By default it assumes that the point lies in the same plane as this
        polygon. If needed, can first check this condition.

        :param point: the point
        :type point: numpy.ndarray *size=3*
        :param check_in_plane: if True, will first check if point and polygon lie in the same plane
        :type check_in_plane: bool
        :param plane_tol: tolerance for check in plane
        :type plane_tol: float
        :return: wether the point is in the plane
        :rtype: bool
        """
        point = point.reshape(3)
        if check_in_plane:
            d = self.get_parametric()[3]
            normal = self.get_normal()
            if not np.allclose(np.dot(normal, point), -d, rtol=plane_tol):
                return False

        matrix = self.get_matrix().T
        projected_polygon = self.project(matrix)
        projected_point = project_points(point, matrix).reshape(3)

        return projected_polygon.get_shapely().intersects(shPoint(projected_point))

    def get_matrix(self, force=False):
        """
        Returns a 3-by-3 orthogonal matrix where is column correspond to an axis of the polygon.
        matrix = [x, y, z] where
        - x belongs to the polygon
        - y belongs to the polygon
        - z is normal to the polygon

        The axes follow the right hand side rule and will be normalized.

        In order to project points into the polygon's coordinates, use the transposed matrix !

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: the matrix of axes
        :rtype: ndarray *shape=(3, 3)*
        """
        if not force and self.matrix is not None:
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
            matrix[i, :] /= norm([matrix[i, :]])

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

    def __init__(self, surface, polyhedra=None, set_of_points=np.empty((0, 3))):
        super().__init__()

        if isinstance(surface, OrientedSurface):
            self.surface = surface
        elif type(surface) == list or type(surface) == np.ndarray:
            self.surface = OrientedSurface(surface)
        else:
            raise ValueError('OrientedPlace needs an OrientedSurface or a numpy.ndarray as surface input')

        if polyhedra is not None:
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

    def obstructs_line_path(self, points):
        """
        Returns whether a line path is obstructed by any polygon present in this place.

        :param points: two points describing line path
        :type points: numpy.ndarray *shape=(2, 3)*
        :return: True if any polygon intercepts the line path
        :rtype: bool
        """
        return polygons_obstruct_line_path(self.get_polygons_iter(), points)

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


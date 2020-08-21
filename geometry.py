import geopandas as gpd
import pyny3d.geoms as pyny
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import modify_class, share_parent_class
from shapely.geometry import Polygon as shPolygon


def enclosed_area(points):
    """
    Returns the enclosed area of the polygon described by the points.
    The polygon is projected on the z=0 plane.
    If results is negative, it means that the curve orientation is ccw.

    :param points: the points of the polygon
    :type points: ndarray *shape=(N, 2 or 3)*
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
    :type points: ndarray *shape=(N, 2 or 3)*
    :return: True if the curve is oriented ccw
    :rtype: bool
    """
    return enclosed_area(points) < 0


def reflexion_on_plane(incidents, normal):
    """
    Return the reflexion of incident vector on a plane with given normal.
    See details: https://en.wikipedia.org/wiki/Reflection_(mathematics)

    :param incidents: incident vectors
    :type incidents: ndarray *shape=(N, 3)*
    :param normal: normal vector to the plane
    :type normal: ndarray *size=3*
    :return: the reflected vector(s)
    :rtype: ndarray *shape(N, 3)*
    """
    normal = normal.reshape((1, 3))
    incidents = incidents.reshape((-1, 3))
    den = normal @ normal.T
    return incidents - (incidents @ normal.T) @ ((2 / den) * normal)  # Order of operation minimizes the # of op.


def project_points(points, matrix, around_point=None):
    """
    Project points on different axes given by matrix columns.

    :param points: the points to be projected
    :type points: ndarray *shape=(N, 3)*
    :param matrix: the matrix to project all the geometry
    :type matrix: ndarray *shape=(3, 3)*
    :param around_point: if present, will apply the project around this point
    :type around_point: ndarray *size=3*
    :return: the new geometry
    :rtype OrientedGeometry
    """
    if around_point is not None:
        around_point = around_point.reshape(1, 3)
        return ((points - around_point) @ matrix.T) + around_point
    else:
        return points @ matrix.T


class OrientedGeometry:
    """
    An oriented geometry object should subclass pyny.root or one of its subclasses.
    It should not add any attribute to the class, only new methods.

    Oriented geometry constructors assume that their input are already oriented.

    This way, casting object between parent and child classes becomes straightforward.
    """
    def superclass(self, inplace=False):
        """
        Parses the object into its parent class.
        If not in place, will return a new object.

        :param inplace: if False, will return a copy
        :type inplace: bool
        :return: the object with type as parent class or nothing
        :rtype: pyny.root or None
        """
        base = self.__class__.__base__
        return modify_class(self, base, inplace=inplace)

    @classmethod
    def cast(cls, obj, inplace=False):
        """
        Parses the object into given subclass or class sharing parent class.
        If not in place, will return a new object.

        :param obj: the object to be parsed
        :type obj: pyny.root
        :param inplace: if False, will return a copy
        :type inplace: bool
        :return: the object with type as subclass or nothing
        :rtype: cls or None
        """
        if not share_parent_class(cls, obj.__class__):
            raise ValueError(f'Cannot cast type {type(obj)} into {cls}.')
        return modify_class(obj, cls, inplace=inplace)

    def project(self, matrix, around_point=None):
        """
        Project a geometry on different axes given by matrix columns.

        :param matrix: the matrix to project all the geometry
        :type matrix: ndarray *shape=(3, 3)*
        :param around_point: if present, will apply the project around this point
        :type around_point: ndarray *size=3*
        :return: the new geometry
        :rtype OrientedGeometry
        """
        raise NotImplementedError

    def plot2d(self, ret=False, ax=None):
        """
        Generates a 2D plot for the z=0 Polygon projection.

        :param ret: if True, returns the figure. It can be used to add
            more elements to the plot or to modify it.
        :type ret: bool
        :param ax: if present, will add content to this ax, otherwise will create a new one
        :type ax: matplotlib axes
        :returns: None, axes
        :rtype: None, matplotlib axes
        """
        raise NotImplementedError


class OrientedPolygon(pyny.Polygon, OrientedGeometry):

    def __init__(self, points):
        super().__init__(points, make_ccw=False)

    def get_matrix(self, normalized=False):
        """
        Returns a 3-by-3 matrix where is column correspond to an axis of the polygon.
        matrix = [x, y, z] where
            x belongs to the polygon
            y belongs to the polygon
            z is normal to the polygon

        The axes follow the right hand side rule.
        Optionally, they can be normalized.

        :param normalized: if True, will normalize each axis
        :type normalized: bool
        :return: the matrix of axes
        :rtype: ndarray *shape=(3, 3)*
        """
        points = self.points
        A = points[0, :]
        B = points[1, :]
        normal = self.get_normal()
        matrix = np.empty((3, 3), dtype=float)
        matrix[0, :] = B - A
        matrix[1, :] = np.cross(normal, matrix[0, :])
        matrix[2, :] = normal

        if normalized:
            for i in range(3):
                matrix[i, :] /= np.linalg.norm([matrix[i, :]])

        return matrix.T

    def project(self, matrix, around_point=None):
        return OrientedPolygon(project_points(self.points, matrix, around_point=around_point))

    def magnify(self, point, factor=None, distance=None):
        if factor is not None:
            pass
        elif distance is not None:
            # https://mathinsight.org/distance_point_plane
            normal = self.get_normal()
            v = point - self.points[0, :]
            d = np.dot(normal, v.reshape(3))
            factor = - distance / d

        else:
            raise ValueError('factor and distance parameters can\'t de None at the same time.')

        point = point.reshape(1, 3)
        vectors = self.points - point
        vectors *= factor
        points = vectors + point
        return OrientedPolygon(points)

    def get_parametric(self, check=True, tolerance=0.001):
        if self.parametric is None:

            # Plane calculation
            normal = np.cross(self.points[1, :] - self.points[0, :],
                               self.points[2, :] - self.points[1, :])
            normal /= np.linalg.norm(normal)  # Normalize
            a, b ,c = normal
            d = -np.dot(np.array([a, b, c]), self.points[2, :])
            self.parametric = np.array([a, b, c, d])

            # Point belonging verification
            if check:
                if self.points.shape[0] > 3:
                    if np.min(np.abs(self.points[3:, 0] * a +
                                     self.points[3:, 1] * b +
                                     self.points[3:, 2] * c +
                                     d)) > tolerance:
                        raise ValueError('Polygon not plane: \n' + \
                                         str(self.points))
        return self.parametric

    def get_normal(self):
        return self.get_parametric(check=False)[:-1]

    def plot2d(self, color='default', alpha=1, ret=False, ax=None, auto_lim=False):
        path = self.get_path()

        if color is 'default':
            color = 'b'

        # Plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.add_patch(patches.PathPatch(path, facecolor=color, lw=1,
                                       edgecolor='k', alpha=alpha))

        if auto_lim:
            domain = self.get_domain()[:, :2]
            ax.set_xlim(domain[0, 0], domain[1, 0])
            ax.set_ylim(domain[0, 1], domain[1, 1])

        if ret:
            return ax


class OrientedSurface(pyny.Surface, OrientedGeometry):

    def __init__(self, polygons, holes=[],
                 melt=False, check_contiguity=False):

        if type(polygons) != list:
            polygons = [polygons]
        if type(holes) != list: holes = [holes]

        if type(polygons[0]) == np.ndarray:
            self.polygons = [OrientedPolygon(polygon)
                             for polygon in polygons]
        elif isinstance(polygons[0], OrientedPolygon):
            self.polygons = polygons
        elif isinstance(polygons[0], pyny.Polygon):
            self.polygons = [
                OrientedPolygon.cast(polygon, inplace=False) for polygon in polygons
            ]
        else:
            raise ValueError('OrientedSurface needs a ndarray, '+\
            'pyny3d.Polygons or OrientedPolygon as input')

        if check_contiguity:
            if not OrientedSurface.contiguous(self.polygons):
                raise ValueError('Non-contiguous polygons in the Surface')

        if len(holes) > 0:
            if type(holes[0]) == np.ndarray:
                self.holes = [OrientedPolygon(hole)
                              for hole in holes]
            elif isinstance(holes[0], OrientedPolygon):
                self.holes = holes
            elif isinstance(holes[0], pyny.Polygon):
                self.holes = [
                    OrientedPolygon.cast(hole, inplace=False) for hole in holes
                ]
        else:
            self.holes = []

        if melt:
            self.melt()

    def project(self, matrix, around_point=None):
        projected_polygons = [polygon.project(matrix, around_point=around_point) for polygon in self.polygons]
        projected_holes = [hole.project(matrix, around_point=around_point) for hole in self.holes]
        return OrientedSurface(projected_polygons, projected_holes)

    def plot2d(self, p_color='default', h_color='w', alpha=1, ret=False, ax=None):
        for polygon in self.polygons:
            ax = polygon.plot2d(color=p_color, alpha=alpha, ret=True, ax=ax)
        for hole in self.holes:
            ax = hole.plot2d(color=h_color, alpha=alpha, ret=True, ax=ax)

        if ret:
            return ax


class OrientedPolyhedron(pyny.Polyhedron, OrientedGeometry):
    """
    A subclass of pyny.Polyhedron with an "inside" and an "outside".
    Surfaces are oriented ccw where watched from the outside. Normal vectors are pointing outward.
    """
    def __init__(self, polygons, **kwargs):
        self.aux_surface = OrientedSurface(polygons, **kwargs)
        self.polygons = self.aux_surface.polygons

    def project(self, matrix, around_point=None):
        surface = self.aux_surface.project(matrix, around_point=around_point)
        return OrientedPolyhedron(surface.polygons)

    def plot2d(self, color='default', alpha=1, ret=False, ax=None):
        for polygon in self.polygons:
            ax = polygon.plot2d(color=color, alpha=alpha, ret=True, ax=ax)

        if ret:
            return ax

    def move(self, d_xyz, inplace=False):
        polygon = np.array([[0, 0], [0, 1], [1, 1], [0, 1]])
        space = OrientedSpace(OrientedPlace(polygon, polyhedra=self))
        if inplace:
            space.move(d_xyz, inplace=inplace)
            self = space.places[0].polyhedra[0]
        else:
            return space.move(d_xyz, inplace=inplace)[0].polyhedra[0]


class Pyramid(OrientedPolyhedron):

    @staticmethod
    def by_point_and_polygon(point, polygon):

        base_points = polygon.points
        polygons = [polygon]

        n = base_points.shape[0]
        A = point

        for i in range(n):
            B = base_points[i - 1, :]
            C = base_points[i, :]
            polygon = OrientedPolygon(np.row_stack([C, B, A]))  # ccw
            polygons.append(polygon)


class Building(OrientedPolyhedron):
    """
    A building is an oriented polyhedron constructed by extruding a polygon in the z direction.
    It consists in 2 flat faces, one for ground and one for rooftop, and as many other vertical faces are there
    are vertices in the original polygon.
    """

    @staticmethod
    def by_polygon_and_height(polygon, height, make_ccw=True):
        """
        Constructs a building from a 3D polygon on the ground.

        :param polygon: 3D polygon
        :type polygon: pyny.Polygon or ndarray *shape=(N, 3)*
        :param height: the height of the building
        :type height: float or int
        :param make_ccw: if True, ensure that polygon is oriented correctly
        :type make_ccw: bool
        :return: a building
        :rtype: Building
        """

        if isinstance(polygon, pyny.Polygon):
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

        if top.get_parametric(check=False)[2] < 0:  # z component should be positive
            top.parametric = - top.parametric
        if bottom.get_parametric(check=False)[2] > 0:  # z component should be negative
            bottom.parametric = - bottom.parametric

        n = top_points.shape[0]

        polygons = [top, bottom]

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
    def by_polygon2d_and_height(polygon, height, make_ccw=True):
        """
        Constructs a building from a 2D polygon.

        :param polygon: 2D polygon
        :type polygon: Polygon (shapely or pyny3d) or ndarray *shape=(N, 3)*
        :param height: the height of the building
        :type height: float or int
        :param make_ccw: if True, ensure that polygon is oriented correctly
        :type make_ccw: bool
        :return: a building
        :rtype: Building
        """

        if isinstance(polygon, pyny.Polygon):
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

        polygon = pyny.Polygon(bottom_points, make_ccw=False)

        return Building.by_polygon_and_height(polygon, height, make_ccw=make_ccw)


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

        Cube.cast(building, inplace=True)  # Cast to Cube object

        return building


class OrientedPlace(pyny.Place, OrientedGeometry):

    def __init__(self, surface, polyhedra=[], set_of_points=np.empty((0, 3)),
                 melt=False):

        if isinstance(surface, OrientedSurface):
            self.surface = surface
        elif isinstance(surface, pyny.Surface):
            self.surface = OrientedSurface(surface.polygons, holes=surface.holes)
        elif type(surface) == dict:  # Seed
            self.surface = OrientedSurface(**surface)
        elif type(surface) == list or type(surface) == np.ndarray:
            self.surface = OrientedSurface(**{'polygons': surface, 'melt': melt})
        else:
            raise ValueError('OrientedPlace needs a dict, OrientedSurface or pyny3d.Surface as input')

        if polyhedra != []:
            if type(polyhedra) != list:
                polyhedra = [polyhedra]
            if isinstance(polyhedra[0], OrientedPolyhedron):
                self.polyhedra = polyhedra
            elif isinstance(polyhedra[0], pyny.Polyhedron):
                self.polyhedra = [
                    OrientedPolyhedron(polyhedron.polygons) for polyhedron in polyhedra
                ]
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

    def project(self, matrix, around_point=None):
        surface = self.surface.project(matrix, around_point=around_point)
        polyhedra = [polyhedron.project(matrix, around_point=around_point) for polyhedron in self.polyhedra]
        set_of_points = project_points(self.set_of_points, matrix, around_point=around_point)
        return OrientedPlace(surface, polyhedra=polyhedra, set_of_points=set_of_points)

    def plot(self, color='default', ret=False, ax=None, normals=False, orientation=False, tight_box=False):
        ax = super().plot(color=color, ret=True, ax=ax)

        if normals or orientation:
            for polyhedron in self.polyhedra:
                for polygon in polyhedron.polygons:
                    x, y, z = polygon.get_centroid()
                    u, v, w, _ = polygon.get_parametric()
                    if normals:
                        ax.quiver(x, y, z, u, v, w, length=4, normalize=True, linewidth=2, color='k')

                    points = polygon.points

                    if orientation:
                        n = points.shape[0]
                        for i in range(n):
                            x, y, z = points[i, :]
                            u, v, w = points[(i + 1) % n, :] - points[i, :]
                            ax.quiver(x, y, z, u, v, w, length=4, normalize=True, linewidth=2, color='r')

        if tight_box:
            domain = self.get_domain()
            ax.set_xlim([domain[0][0], domain[1][0]])
            ax.set_ylim([domain[0][1], domain[1][1]])
            ax.set_zlim([domain[0][2], domain[1][2]])

        if ret:
            return ax

    def plot2d(self, poly_color='default', h_color='w', point_color='r', alpha=1, ret=False, ax=None):
        ax = self.surface.plot2d(p_color=poly_color, h_color=h_color, alpha=alpha, ax=ax, ret=True)

        for polyhedron in self.polyhedra:
            ax = polyhedron.plot2d(color=poly_color, alpha=alpha, ax=ax, ret=True)

        if self.set_of_points.size > 0:
            points = self.set_of_points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=point_color)

        domain = self.get_domain()[:, :2]
        ax.set_xlim(domain[0, 0], domain[1, 0])
        ax.set_ylim(domain[0, 1], domain[1, 1])

        if ret:
            return ax


class OrientedSpace(pyny.Space, OrientedGeometry):

    def __init__(self, places=[]):
        # Empty initializations
        self.places = []

        # Lock attributes
        self.locked = False
        self.map = None
        self.seed = None
        self.map2seed_schedule = None
        self.explode_map_schedule = None

        # Creating the object
        if places != []:
            if type(places) != list:
                places = [places]
            if isinstance(places[0], OrientedPlace):
                self.add_places(places)
            elif type(places[0]) == dict:
                # Initialize the places
                self.add_places([
                    OrientedPlace(**place) for place in places
                ])
            else:
                raise ValueError('OrientedSpace needs a list, dict, '+\
                                 'pyny3d.Place or OrientedPlace as input')

    def project(self, matrix, around_point=None):
        return OrientedSpace(places=[place.project(matrix, around_point=around_point) for place in self.places])

    def plot2d(self, p_color='default', h_color='w', alpha=1, ret=False, ax=None):

        for place in self.places:
            ax = place.plot2d(p_color=p_color, h_color=h_color, alpha=alpha, ax=ax, ret=True)

        if ret:
            return ax


def generate_place_from_rooftops_file(roof_top_file, center=True):
    gdf = gpd.read_file(roof_top_file)
    gdf.dropna(subset=['height'], inplace=True)
    gdf.to_crs(epsg=3035, inplace=True)

    if center:
        bounds = gdf.total_bounds
        x = (bounds[0] + bounds[2]) / 2
        y = (bounds[1] + bounds[3]) / 2

        gdf['geometry'] = gdf['geometry'].translate(-x, -y)

    def func(series: gpd.GeoSeries):
        return Building.by_polygon2d_and_height(series['geometry'], series['height'])

    polyhedra = gdf.apply(func, axis=1).values.tolist()

    bounds = gdf.total_bounds.reshape(2, 2)

    points = np.empty((4, 2))
    points[0::3, 0] = bounds[0, 0]
    points[1:3, 0] = bounds[1, 0]
    points[:2, 1] = bounds[0, 1]
    points[2:, 1] = bounds[1, 1]

    ground_surface = pyny.Surface(points)

    place = OrientedPlace(ground_surface)
    place.polyhedra = polyhedra

    return place


if __name__ == '__main__':

    #set_poly_verify(False)

    place = generate_place_from_rooftops_file('data/small.geojson')

    domain = place.get_domain()
    #ground = buildings.surface
    ground_center = place.get_centroid()

    tx = ground_center + [5, 5, 1]
    rx = ground_center + [-2, 0, 5]
    tx = tx.reshape(1, 3)
    rx = rx.reshape(1, 3)
    cube = Cube.by_point_and_side_length(tx, 2)

    from scipy.spatial.transform import Rotation as R

    rot2 = R.from_euler('xyz', [0, 20, 0], degrees=True).as_matrix()

    cube = cube.project(rot2, around_point=tx)

    place.polyhedra.append(
        cube
    )

    place.add_set_of_points(tx)
    place.add_set_of_points(rx)
    ax = place.iplot(ret=True)
    print(place.polyhedra)

    #ax.scatter(tx[0], tx[1], tx[2])
    #ax.scatter(rx[0], rx[1], rx[2])


    face = cube.polygons[2]

    face.plot(color='b', ax=ax)

    face_bis = face.magnify(tx, distance=10)

    face_bis.plot(color='g', ax=ax)


    matrix = face.get_matrix(normalized=True)

    print(matrix)
    S = place.project(matrix)

    S.plot2d(alpha=1)

    #face_bis.project(matrix).plot2d(alpha=0.2, color='r')

    plt.show()

    """
    for polyhedron in S.polyhedra:
        polyhedron.aux_surface.plot2d()
        plt.show()
    """





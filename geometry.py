import geopandas as gpd
import pyny3d.geoms as pyny
import numpy as np
import matplotlib.pyplot as plt
import copy


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


def project_polygon(space, point, plane):
    pass


def modify_class(obj, new_class, inplace=False):
    if inplace:
        obj.__class__ = new_class
    else:
        obj_copy = copy.deepcopy(obj)
        obj_copy.__class__ = new_class
        return obj_copy


class OrientedGeometry:

    def superclass(self, inplace=False):
        base = self.__class__.__base__
        return modify_class(self, base, inplace=inplace)

    @classmethod
    def cast(cls, obj, inplace=False):
        return modify_class(obj, cls, inplace=inplace)


class OrientedPolygon(pyny.Polygon, OrientedGeometry):

    def get_parametric(self, check=True, tolerance=0.001):
        if self.parametric is None:

            # Plane calculation
            a, b, c = np.cross(self.points[1, :] - self.points[0, :],
                               self.points[2, :] - self.points[1, :])
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


class OrientedSurface(pyny.Surface, OrientedGeometry):

    def __init__(self, polygons, *args, **kwargs):

        if type(polygons) != list:
            polygons = [polygons]

        for polygon in polygons:
            if isinstance(polygon, OrientedPolygon):
                polygon.superclass(inplace=True)

        super().__init__(polygons, *args, **kwargs)

        for polygon in self.polygons:
            OrientedPolygon.cast(polygon, inplace=True)


class OrientedPolyhedron(pyny.Polyhedron, OrientedGeometry):
    """
    A subclass of pyny.Polyhedron with an "inside" and an "outside".
    Surfaces are oriented ccw where watched from the outside. Normal vectors are pointing outward.
    """
    def __init__(self, polygons, **kwargs):
        self.aux_surface = OrientedSurface(polygons, **kwargs)
        self.polygons = self.aux_surface.polygons

    def move(self, d_xyz, inplace=False):
        polygon = np.array([[0, 0], [0, 1], [1, 1], [0, 1]])
        space = OrientedSpace(OrientedPlace(polygon, polyhedra=self))
        if inplace:
            space.move(d_xyz, inplace=inplace)
            self = space.places[0].polyhedra[0]
        else:
            return space.move(d_xyz, inplace=inplace)[0].polyhedra[0]


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
        top_points = np.array([
            [r, r, r],
            [-r, r, r],
            [-r, -r, r],
            [r, -r, r]
        ])

        bottom_points = np.array(top_points[::-1, :])  # Copy
        bottom_points[:, -1].fill(-r)

        top_polygon = OrientedPolygon(top_points)
        bottom_polygon = OrientedPolygon(bottom_points)

        polygons = [top_polygon, bottom_polygon]

        # Create 4 remaining faces
        for i in range(4):
            A = top_points[i - 1, :]
            B = bottom_points[i - 1, :]
            C = bottom_points[i, :]
            D = top_points[i, :]
            polygon = OrientedPolygon(np.row_stack([A, B, C, D]))
            polygons.append(polygon)

        state = pyny.Polygon.verify
        pyny.Polygon.verify = False  # Required because Polyhedron.move function will throw error otherwise

        polyhedron = OrientedPolyhedron(polygons).move(point)
        pyny.Polygon.verify = state  # Reset to previous state

        return Cube(polyhedron.polygons)  # Cast to Cube object


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
            polygon = pyny.Polygon(np.row_stack([C, B, A]))  # ccw
            polygons.append(polygon)


class Building(OrientedPolyhedron):
    """
    A building is an oriented polyhedron constructed by extruding a polygon in the z direction.
    It consists in 2 flat faces, one for ground and one for rooftop, and as many other vertical faces are there
    are vertices in the original polygon.
    """
    @staticmethod
    def by_polygon2d_and_height(polygon, height, make_ccw=True):
        """
        Constructs a building from a 2D polygon
        :param polygon: 2D polygon
        :type polygon: Polygon (shapely or pyny3d)
        :param height: the height of the building
        :type height: float or int
        :param make_ccw: if True, ensure that polygon is oriented correctly
        :type make_ccw: bool
        :return: a building
        :rtype: Building
        """

        if isinstance(polygon, pyny.Polygon):
            polygon = polygon.shapely()

        x, y = polygon.exterior.coords.xy
        x = x[:-1]
        y = y[:-1]
        z = np.full_like(x, height)
        z0 = np.zeros_like(x)
        top_points = np.column_stack([x, y, z])
        bottom_points = np.column_stack([x, y, z0])

        if make_ccw:
            if not is_ccw(top_points):
                top_points = top_points[::-1, :]
            else:
                bottom_points = bottom_points[::-1, :]

        top = OrientedPolygon(top_points, make_ccw=False)
        bottom = OrientedPolygon(bottom_points, make_ccw=False)

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

            polygon = OrientedPolygon(face_points, make_ccw=False)
            polygons.append(polygon)
            """
            vertex_1 = B - A
            vertex_2 = C - A
            vertex_3 = np.cross(vertex_2, vertex_1)

            normal = polygon.get_parametric(check=False)[:-1]

            dot = np.dot(vertex_3, normal)

            if dot < 0:
                polygon.parametric = - polygon.parametric
            """

        return Building(polygons, make_ccw=False)


class OrientedPlace(pyny.Place, OrientedGeometry):

    def __init__(self, surface, polyhedra=[], **kwargs):
        if isinstance(surface, OrientedSurface):
            surface.superclass(inplace=True)
        if type(polyhedra) != list:
            polyhedra = [polyhedra]

        if isinstance(polyhedra[0], OrientedPolyhedron):
            for polyhedron in polyhedra:
                polyhedron.superclass(inplace=True)

        super().__init__(surface, polyhedra=polyhedra, **kwargs)

        OrientedSurface.cast(self.surface, inplace=True)

        for polyhedron in self.polyhedra:
            OrientedPolyhedron.cast(polyhedron, inplace=True)


class OrientedSpace(pyny.Space, OrientedGeometry):

    def __init__(self, places=[], **kwargs):
        if type(places) != list:
            places = [places]
        if isinstance(places[0], OrientedPlace):
            for place in places:
                place.superclass(inplace=True)
        super().__init__(places=places, **kwargs)

        for place in self.places:
            OrientedPlace.cast(place, inplace=True)


def generate_place_from_rooftops_file(roof_top_file):
    gdf = gpd.read_file(roof_top_file)
    gdf.dropna(subset=['height'], inplace=True)
    gdf.to_crs(epsg=3035, inplace=True)

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

    place = pyny.Place(ground_surface)
    place.polyhedra = polyhedra

    return place


if __name__ == '__main__':

    #set_poly_verify(False)

    place = generate_place_from_rooftops_file('data/small.geojson')

    domain = place.get_domain()
    #ground = buildings.surface
    ground_center = place.get_centroid()

    S = place
    ax = S.iplot(ret=True)


    tx = ground_center + [5, 5, 2]
    rx = ground_center + [-2, 0, 5]
    ax.scatter(tx[0], tx[1], tx[2])
    ax.scatter(rx[0], rx[1], rx[2])



    place.polyhedra.append(
        Cube.by_point_and_side_length(tx, 5)
    )

    for polyhedron in place.polyhedra:
        for polygon in polyhedron.polygons:
            x, y, z = polygon.get_centroid()
            u, v, w, _ = polygon.get_parametric()
            ax.quiver(x, y, z, u, v, w, length=4, normalize=True, linewidth=2, color='b')

            points = polygon.points

            #print('Is ccw:', polygon.is_ccw())

            n = points.shape[0]

            for i in range(n):
                x, y, z = points[i, :]
                u, v, w = points[(i + 1) % n, :] - points[i, :]
                #ax.quiver(x, y, z, u, v, w, length=4, normalize=True, linewidth=2, color='r')

    plt.xlim([domain[0][0], domain[1][0]])
    plt.ylim([domain[0][1], domain[1][1]])
    ax.set_zlim([domain[0][2], domain[1][2]])
    plt.show()




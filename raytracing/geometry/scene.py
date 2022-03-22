import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon as shPolygon

from ..plotting import Plotable
from .polygon import Polygon
from .polyhedron import Polyhedron


class Scene(Plotable):
    def __init__(self, geometries):
        super().__init__()
        self.geometries = geometries
        self.paths = []

        domains = np.dstack([geometry.domain for geometry in geometries])
        self.domain = np.zeros((2, 3), dtype=float)
        self.domain[0, :] = domains[0, :, :].min(axis=-1)
        self.domain[1, :] = domains[1, :, :].max(axis=-1)

    def plot(self):
        for geometry in self.geometries:
            geometry.on(self.ax).plot()

        self.set_limits(self.domain)

        return self.ax

    @staticmethod
    def from_geojson(
        filename, drop_missing_heights=True, default_height=10, center=True
    ):
        gdf = gpd.read_file(filename)

        # Only keeping polygons (sometimes points are given)
        gdf = gdf[[isinstance(g, shPolygon) for g in gdf["geometry"]]]

        if drop_missing_heights:
            gdf.dropna(subset=["height"], inplace=True)
        else:
            if "height" not in gdf:
                gdf["height"] = default_height
            else:
                gdf["height"].fillna(value=default_height, inplace=True)

        gdf.to_crs(
            epsg=3035, inplace=True
        )  # To make buildings look more realistic, there may be a better choice :)

        if center:
            bounds = gdf.total_bounds
            x = (bounds[0] + bounds[2]) / 2
            y = (bounds[1] + bounds[3]) / 2

            gdf["geometry"] = gdf["geometry"].translate(-x, -y)

        def func(series):
            return Polyhedron.from_2d_polygon(
                series["geometry"], height=series["height"], keep_ground=False
            )

        polyhedra = gdf.apply(func, axis=1).values.tolist()

        bounds = gdf.total_bounds.reshape(2, 2)

        points = np.zeros((4, 3), dtype=float)
        points[0::3, 0] = bounds[0, 0]
        points[1:3, 0] = bounds[1, 0]
        points[:2, 1] = bounds[0, 1]
        points[2:, 1] = bounds[1, 1]

        ground_surface = Polygon(points)

        return Scene([ground_surface, *polyhedra])

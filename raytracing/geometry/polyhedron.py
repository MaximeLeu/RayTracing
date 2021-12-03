from base import Geometry


class Polyhedron(Geometry):

    def __init__(self, polygons, **kwargs):
        super().__init__(**kwargs)

        self.polygons = polygons


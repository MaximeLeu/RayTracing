class Edge(object):
    """
    Edge on which diffraction can occur.
    """

    def tangent(self, x, y, z):
        raise NotImplementedError

    def contains(self, x, y, z):
        raise NotImplementedError

    def s_to_xyz(self, s):
        raise NotImplementedError

class Surface(object):
    """
    Surface on which specular reflection can occur.
    """

    def normal(self, x, y, z):
        raise NotImplementedError

    def contains(self, x, y, z):
        raise NotImplementedError

    def st_to_xyz(self, s, t):
        raise NotImplementedError

    def reflection(self, i, x, y, z):
        n = self.norma(x, y, z)
        return i - 2 * np.dot(i, n) * n

import radarcoverage.geometry as geom
import numpy as np


def np_normalize_path(points):
    vectors = np.diff(points, axis=0)
    n = np.linalg.norm(vectors, axis=1)
    return vectors / n.reshape(-1, 1), n


def np_intersection_of_2d_lines(points_A, points_B):
    s = np.vstack([points_A, points_B])
    h = np.hstack([s, np.ones((4, 1))])
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return np.full(2, np.nan)
    else:
        return np.array([x / z, y / z])


def np__gamma__(v):
    norms = np.linalg.norm(v, axis=1)
    return (norms[:-1] / norms[1:]).reshape(-1, 1)


class TestGeometry(np.testing.TestCase):

    def test_normalize_path(self):

        my_func = geom.normalize_path
        np_func = np_normalize_path

        args = [
            np.random.rand(10, 3)
        ]

        for i, arg in enumerate(args):
            with self.subTest(i=i):
                got_path, got_n = my_func(arg)
                expected_path, expected_n = np_func(arg)

                np.testing.assert_almost_equal(got_path, expected_path)
                np.testing.assert_almost_equal(got_n, expected_n)

    def test_intersection_of_2d_lines(self):
        my_func = geom.intersection_of_2d_lines
        np_func = np_intersection_of_2d_lines

        args = [
            (np.random.rand(2, 2), np.random.rand(2, 2))
        ]

        for i, arg in enumerate(args):
            with self.subTest(i=i):
                got = my_func(*arg)
                expected = np_func(*arg)

                np.testing.assert_almost_equal(got, expected)

    def test__gamma__(self):
        my_func = geom.__gamma__
        np_func = np__gamma__

        args = [
            np.random.rand(10, 3)
        ]

        for i, arg in enumerate(args):
            with self.subTest(i=i):
                got = my_func(arg)
                expected = np_func(arg)

                np.testing.assert_almost_equal(got, expected)


if __name__ == '__main__':
    np.testing.main()

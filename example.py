if __name__ == "__main__":
    import numpy as np

    from raytracing import Polygon, Polyhedron

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )

    p1 = Polygon([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5]])
    p2 = Polygon([[0, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]])
    p3 = Polygon([[1, 0, 0], [1, 1, 0], [0.5, 0.5, 0.5]])
    p4 = Polygon([[1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5]])
    p5 = Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    ph = Polyhedron([p1, p2, p3, p4, p5])
    ph.plot()
    ph.show()

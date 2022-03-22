if __name__ == "__main__":
    import numpy as np

    from raytracing import Polygon

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    p = Polygon(points)
    p.plot()
    p.show()

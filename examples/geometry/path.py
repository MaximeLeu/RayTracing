if __name__ == "__main__":
    import numpy as np

    from raytracing import Path

    points = np.array([[0, 0, 0], [1, 1, 1], [2, 4, 9]])

    path = Path(points)
    path.plot()
    path.show()

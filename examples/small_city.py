if __name__ == "__main__":
    from itertools import zip_longest

    import numpy as np

    from raytracing import Point, Scene, Simulation

    filename = "~/repositories/RayTracing/data/ny_60th_street.geojson"
    filename = "~/Downloads/export (1).geojson"

    scene = Scene.from_geojson(filename)

    start_1 = np.array([-130, -34, 1.5])
    start_1 = np.array([-130, -34, 1.5])

    RXS = []

    for x, y, z in zip_longest(
        np.linspace(-10, 41, 12), np.linspace(6, -1.7, 12), [1.5], fillvalue=1.5
    ):
        RXS.append(Point([x, y, z]))

    for x, y, z in zip_longest(
        np.linspace(-33, 43, 12), np.linspace(27, 12, 12), [1.5], fillvalue=1.5
    ):
        RXS.append(Point([x, y, z]))

    TX = Point([-16.5, -7.4, 10])
    # RX = Point([-20, 20, 2])
    RXS = [
            Point([-21.6, -9.7, 1.5]),
            Point([-31.3, -12.7, 1.5]),
            Point([-11.6, -5.4, 1.5]),
            Point([-3.4, -1.8, 1.5]),
            Point([8.2, 6.4, 1.5]),
            Point([18.2, 18.2, 1.5]),
            Point([26, 26, 1.5]),
            Point([30.7, 26, 1.5]),
    ]

    simu = Simulation(scene, TX, *RXS)

    simu.compute_paths(3)
    simu.save_paths("paths_ny.pickle")

    import pickle

    with open("broadway_simu.pickle", "wb") as f:
        pickle.dump(simu, f)

    simu.plot()
    simu.show()

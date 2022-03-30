if __name__ == "__main__":
    from itertools import zip_longest

    import numpy as np

    from raytracing import Point, Scene, Simulation

    filename = "~/repositories/RayTracing/data/broadway.geojson"

    scene = Scene.from_geojson(filename)

    start_1 = np.array([-130, -34, 1.5])
    start_1 = np.array([-130, -34, 1.5])

    RXS = []

    for x, y, z in zip_longest(
        np.linspace(-130, 77, 25), np.linspace(-34, 38, 25), [1.5], fillvalue=1.5
    ):
        RXS.append(Point([x, y, z]))

    for x, y, z in zip_longest(
        np.linspace(4.2, 38, 25), np.linspace(-71, 144, 25), [1.5], fillvalue=1.5
    ):
        RXS.append(Point([x, y, z]))

    TX = Point([110, 21, 90])
    # RX = Point([-20, 20, 2])

    simu = Simulation(scene, TX, *RXS)

    simu.compute_paths(3)
    simu.save_paths("paths_broadway.pickle")

    import pickle

    with open("broadway_simu.pickle", "wb") as f:
        pickle.dump(simu, f)

    simu.plot()
    simu.show()

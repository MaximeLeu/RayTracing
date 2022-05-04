if __name__ == "__main__":
    from itertools import zip_longest

    import numpy as np

    from raytracing import Point, Scene, Simulation, Polyhedron, Polygon, Square

    ground = Polygon([[-20, -20, 0], [60, -20, 0], [60, 20, 0], [-20, 20, 0]])
    ground.edges = []

    b1_base = Polygon([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]])

    translate = b1_base.points[:, :-1] * 0
    translate[:, 0] = 1

    b1 = Polyhedron.from_2d_polygon(
        b1_base.points[:, :-1], height=20, keep_ground=False
    )
    b2 = Polyhedron.from_2d_polygon(
        b1_base.points[:, :-1] + translate * 30, height=40, keep_ground=False
    )
    b3 = Polyhedron.from_2d_polygon(
        b1_base.points[:, :-1] + translate * 15, height=10, keep_ground=False
    )

    scene = Scene([ground, b1, b2, b3])

    TX = Point([0, 0, 22])
    RXS = [
        Point([8, 0, 2]),
    ]

    simu = Simulation(scene, TX, *RXS)

    simu.compute_paths(3)
    simu.save_paths("010.pickle")

    import pickle

    with open("010_simu.pickle", "wb") as f:
        pickle.dump(simu, f)

    simu.plot()
    simu.show()

if __name__ == "__main__":
    from raytracing import Point, Scene, Simulation

    filename = "~/repositories/RayTracing/data/manhattan.geojson"

    scene = Scene.from_geojson(filename)

    TX = Point([20, 20, 10])
    RX = Point([-20, 20, 2])

    simu = Simulation(scene, TX, RX)

    # simu.compute_paths(0)

    simu.plot()
    simu.show()

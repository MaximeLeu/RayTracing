if __name__ == "__main__":
    from raytracing import Point, Scene, Simulation

    filename = "~/repositories/RayTracing/data/ny.geojson"

    scene = Scene.from_geojson(filename)

    TX = Point([0, 0, 0])
    RX = Point([10, 10, 4])

    # scene.add_object(...)

    # scene.trace_paths(BS=TX, to=RX)

    simu = Simulation(scene, TX, RX)

    simu.plot()
    simu.show()

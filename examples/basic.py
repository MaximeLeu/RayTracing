if __name__ == "__main__":
    from raytracing import Point, Polyhedron, Scene, Simulation

    filename = "~/repositories/RayTracing/data/ny.geojson"
    filename = "~/repositories/RayTracing/data/small.geojson"

    scene = Scene.from_geojson(filename)
    polyhedron = Polyhedron.from_2d_polygon(
        [[-10, -10], [10, -10], [-10, 10]], height=30
    )
    # scene = Scene([polyhedron])

    TX = Point([20, 20, 10])
    RX = Point([-20, 20, 2])

    # scene.add_object(...)

    # scene.trace_paths(BS=TX, to=RX)

    simu = Simulation(scene, TX, RX)

    simu.compute_paths(2)

    simu.plot()
    simu.show()

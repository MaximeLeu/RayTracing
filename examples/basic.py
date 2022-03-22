if __name__ == "__main__":
    from raytracing import Scene

    filename = "~/repositories/RayTracing/data/ny.geojson"

    scene = Scene.from_geojson(filename)

    # scene.add_object(...)

    # scene.trace_paths(BS=TX, to=RX)

    scene.plot()
    scene.show()

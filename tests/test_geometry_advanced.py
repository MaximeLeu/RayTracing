#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:29:18 2023
@author: Maxime Leurquin
Testing of functions defined in the geometry file.
"""
import numpy as np
import raytracing.geometry as geom
from raytracing.place_utils import create_small_place
from raytracing.ray_tracing import RayTracingProblem


def test_all_save_load():
    
    def test_save_load_oriented_polygon(oriented_polygon):
        filename="saved_oriented_polygon.json"
        oriented_polygon.to_json(filename)
        loaded_polygon=geom.OrientedPolygon.from_json(filename=filename)
        assert oriented_polygon==loaded_polygon
        print("save-load oriented polygon SUCCESS")
        return

    def test_save_load_oriented_polyhedron(oriented_polyhedron):
        filename="saved_oriented_polyhedron.json"
        oriented_polyhedron.to_json(filename)
        loaded_polyhedron=geom.OrientedPolyhedron.from_json(filename=filename)
        assert oriented_polyhedron==loaded_polyhedron, f"original {oriented_polyhedron}, loaded {loaded_polyhedron}"
        print("save-load oriented polyhedron SUCCESS")
        return

    def test_save_load_oriented_surface(oriented_surface):
        filename="saved_oriented_polygon.json"
        oriented_surface.to_json(filename)
        loaded_surface=geom.OrientedSurface.from_json(filename=filename)
        assert oriented_surface==loaded_surface
        print("save-load oriented surface SUCCESS")
        return


    def test_save_load_oriented_place(oriented_place):
        filename="saved_oriented_place.json"
        oriented_place.to_json(filename)
        loaded_place=geom.OrientedPlace.from_json(filename=filename)
        assert oriented_place==loaded_place
        print("save-load oriented place SUCCESS")
        return
    
    place,tx,geometry=create_small_place(npoints=3)

    for oriented_polyhedron in place.polyhedra:
        test_save_load_oriented_polyhedron(oriented_polyhedron)
        for oriented_polygon in oriented_polyhedron.polygons:
            test_save_load_oriented_polygon(oriented_polygon)
    
    test_save_load_oriented_surface(place.surface)
    test_save_load_oriented_place(place)
    print("All save-load test before solve SUCCESS")   
    
    
    problem=RayTracingProblem(tx, place)
    idx=np.array(range(len(place.set_of_points)))
    problem.solve(max_order=2,receivers_indexs=idx)
    
    for oriented_polyhedron in place.polyhedra:
        test_save_load_oriented_polyhedron(oriented_polyhedron)
        for oriented_polygon in oriented_polyhedron.polygons:
            test_save_load_oriented_polygon(oriented_polygon)
    
    test_save_load_oriented_surface(place.surface)
    test_save_load_oriented_place(place)
    print("All save-load after solve SUCCESS")   
    return



def test_contains_point():
    #test OrientedPolygon.contains_point
    pass

def test_points_contained():
    #test OrientedPolygon.points_containes
    pass

def test_rotate():
    #test OrientedPolygon.rotate
    pass

def test_split():
    #test OrientedPolygon.split
    pass


def test_overlap():
    #test OrientedPolyhedron.overlap
    pass

def test_extend_polyhedron():
    #test OrientedPolyhedron.extend_polyhedron
    pass


def test_overlap_place():
    #test OrientedPlace.overlap_place
    pass


def test_add_tree():
    #test OrientedPlace.add_tree
    pass


def test_generate_place_from_rooftops_file():
    pass

def preprocess_geojson():
    pass


def test_building_on_slope():
    pass

def test_rebuild_building():
    pass


if __name__ == '__main__':
    # place,tx,geometry=create_small_place(npoints=3)
    # problem=RayTracingProblem(tx, place)
    # idx=np.array(range(len(place.set_of_points)))
    # problem.solve(max_order=2,receivers_indexs=idx)
    
    # solved_pb_sharp_edges=problem.place.sharp_edges
    # solved_place_sharp_edges=place.sharp_edges


    # filename="saved_oriented_place.json"
    # place.to_json(filename)
    # loaded_place=geom.OrientedPlace.from_json(filename=filename)
    # loaded_place_sharp_edges=loaded_place.sharp_edges
    
    # assert(solved_place_sharp_edges==loaded_place_sharp_edges)
    
    test_all_save_load()
    
    
   
    
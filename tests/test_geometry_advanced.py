#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:29:18 2023
@author: Maxime Leurquin
Testing of functions defined in the geometry file.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as shPolygon
import raytracing.geometry as geom
from raytracing.ray_tracing import RayTracingProblem
from raytracing.materials_properties import set_properties

import raytracing.file_utils as file_utils
import raytracing.plot_utils as plot_utils
import raytracing.place_utils as place_utils

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
    
    place,tx,geometry=place_utils.create_small_place(npoints=3)

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


def test_normalize_path():
    points=np.array([[0,0,0],
                    [1,1,1],
                    [3,3,3]])
    si,Si,sr,Sr=geom.normalize_path(points)
    assert np.all(si==np.array([1,1,1])/np.sqrt(3))
    assert np.all(sr==np.array([2,2,2])/np.sqrt(12))
    assert Si==np.sqrt(3) and Sr==np.sqrt(12)
    
    
    points=np.array([[1,2,3],
                    [4,4,4],
                    [10,11,12]])
    si,Si,sr,Sr=geom.normalize_path(points)
    assert np.all(si==np.array([3,2,1])/np.sqrt(14))
    assert np.all(sr==np.array([6,7,8])/np.sqrt(36+49+64))
    assert Si==np.sqrt(14) and Sr==np.sqrt(36+49+64)
    print("normalize_path test success")
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
    geometry_filename='../data/small.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    pass

def test_preprocess_geojson():
    
    pass


def test_by_polygon2d_and_height():
    points1=np.array([[1,2],
                     [2,2],
                     [3,4],
                     [3,4]])
    polyhedron1=geom.Building.by_polygon2d_and_height(points1,height=10)
    
    points2=np.array([[5,10],
                     [6,12],
                     [7,14],
                     [8,16]])
    polyhedron2=geom.Building.by_polygon2d_and_height(shPolygon(points2),height=10)


    polyhedrons=[polyhedron1,polyhedron2]
    
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for polyhedron in polyhedrons:
        ax=polyhedron.plot3d(ax=ax,ret=True)
    
    return



def test_building_on_slope():
    geometry="test_building_on_slope"
    init_elevation=10
    alpha=30
    building_height=10
    #create ground
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, init_elevation], [70, 24, init_elevation]]))
    ground=ground.rotate(axis=np.array([1, 0, 0]), angle_deg=180) #othewise the normal is wrongly oriented
    ground=ground.rotate(axis=np.array([0, 1, 0]), angle_deg=alpha)
    
    #create buildings
    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, init_elevation], [17, 24, init_elevation]]))
    building_1 = geom.Building.building_on_slope(square_1, ground,building_height)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, init_elevation], [37, 2, init_elevation]]))
    building_2 = geom.Building.building_on_slope(square_2, ground,building_height)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, init_elevation], [57, 24, init_elevation]]))
    building_3 = geom.Building.building_on_slope(square_3, ground,building_height)
    
    ground.properties=set_properties("ground")
    buildings=[building_1, building_2, building_3]
    for building in buildings:
        building.building_type="appartments"
        building.apply_properties_to_polygons()
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground),buildings)
    #add TX and RX
    tx = np.array([5., 12., init_elevation+10]).reshape(-1, 3)
    rx = np.array([65., 12.,init_elevation+ 10.]).reshape(-1, 3)
    place.add_set_of_points(rx)
    return place, tx, geometry


def test_rebuild_building():
    pass

def create_squares_test_split():
    # Define corner points for the first rectangle
    points1 = np.array([[0, 0, -5],
                       [10, 5, 0]])
    rectangle1 = geom.Square.by_2_corner_points(points1)
    rectangle1=rectangle1.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    points2 = np.array([[10, 0, 0],
                      [20, 5, -5]])
    rectangle2 = geom.Square.by_2_corner_points(points2)
    rectangle2=rectangle2.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    points3 = np.array([[0, 5, 0],
                      [20, 10, 0]])
    rectangle3 = geom.Square.by_2_corner_points(points3)
    rectangle3=rectangle3.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    middle_square=geom.Square.by_center_and_side(center=[10,5,0],side=4)
    middle_square=middle_square.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    rectangles=[rectangle1,rectangle2,rectangle3]
    fig = plt.figure("the test")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax=plot_utils.ensure_axis_orthonormal(ax)
    for rectangle in rectangles:
        rectangle.plot3d(facecolor=(0,1,1,1),ax=ax,alpha=1,normal=True)
        rectangle.properties = set_properties("ground")
            
    middle_square.plot3d(facecolor=(1,1,1,1),ax=ax,alpha=1,normal=True)
    plt.show()
    return rectangles,middle_square


def test_split():
    #unzoom the figures to see something.
    tx = np.array([5, 12, 10]).reshape(-1, 3)
    #create grounds
    rectangles,middle_square=create_squares_test_split()
    #split the square on the ground boundaries
    splits=middle_square.split(rectangles)
    
    fig = plt.figure("splitted rectangles")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax=plot_utils.ensure_axis_orthonormal(ax)
    for split in splits:
        split.plot3d(facecolor=(0,1,1,1),ax=ax,alpha=1,normal=True)
        assert len(split)==4,'square made of more than 4 points'
    fig.show()
    
    buildings=[]
    for i,split in enumerate(splits):
         buildings.append(geom.Building.building_on_slope(split, rectangles[i],height=10))
         print(f"SPLIT: {split}")
    #create place
    fig.show()
    place = geom.OrientedPlace(rectangles,buildings)
    place_utils.plot_place(place,tx,show_normals=True)
    return
   




if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    # place,tx,geometry=place_utils.create_small_place(npoints=3)
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
    
    #test_all_save_load()
    test_normalize_path()
    test_by_polygon2d_and_height()
    
    plt.close("all")
    place,tx,geometry=test_building_on_slope()
    test_split()
    
   
    
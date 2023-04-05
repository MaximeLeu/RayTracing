#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:44:15 2023

@author: Maxime Leurquin
"""
#packages
import numpy as np
import matplotlib.pyplot as plt



#self written imports
import raytracing.geometry as geom
from raytracing.materials_properties import set_properties, LAMBDA

import raytracing.plot_utils as plot_utils
import raytracing.place_utils as place_utils
import raytracing.file_utils as file_utils
file_utils.chdir_to_file_dir(__file__)



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
    #save
    place.to_json(filename=f"../tests/{geometry}.json")
    return place, tx, geometry


def test_split():
    #unzoom the figures to see something.
    tx = np.array([5, 12, 10]).reshape(-1, 3)
    #create grounds
    rectangles,middle_square=test_square()
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
   
def test_square():
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



if __name__ == '__main__':
    plt.close("all")
    place,tx,geometry=test_building_on_slope()
    test_split()
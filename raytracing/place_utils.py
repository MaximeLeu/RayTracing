#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:52:10 2022

File containing the functions used to create the places.
@author: max
"""
#self written imports
import raytracing.geometry as geom
import plot_utils
from materials_properties import set_properties

#packages
import numpy as np
import matplotlib.pyplot as plt


def plot_place(place,tx):
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")
    place.center_3d_plot(ax)   
    ax = place.plot3d(ax=ax)
   
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001) 


def create_small_place(npoints=3):
    geometry="small"
    #create place
    geometry_filename='../data/small.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    #add TX and RX   
    tx = np.array([3, 38, 18]).reshape(-1, 3)
    rx=np.array([0,25,1.5]).reshape(-1,3)
    for i in range(npoints):
        place.add_set_of_points(rx)
        rx =rx+np.array([0,-4,0])    
    #save and plot    
    place.to_json(filename="../data/small.json")
    plot_place(place,tx)
    return place,tx,geometry


def create_levant_place(npoints=15):
    geometry="levant"
    #create place
    geometry_filename='../data/place_levant.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    #constants required for TX and RX positions
    ST_BARBE_COORD=np.array([-15,-46,0])
    MAXWELL_COORDINATES=np.array([84,-46,0]) #careful not to put this point inside the maxwell
    MAXWELL_HEIGHT=40 #find it in the geojson
    RX_HEIGHT=1.2
    TX_HEIGHT=3
    #add TX and RX
    tx=(MAXWELL_COORDINATES+[5,0,MAXWELL_HEIGHT+TX_HEIGHT]).reshape(-1, 3)
    rx0=(MAXWELL_COORDINATES+[-25,0,RX_HEIGHT]).reshape(-1,3)  
    dist=np.linalg.norm(rx0-ST_BARBE_COORD)
    step=dist/npoints
    for receiver in range(npoints):
        rx =rx0+np.array([-receiver*step,0,0])
        place.add_set_of_points(rx)
    #plot
    print(f"MAXWELL: {MAXWELL_COORDINATES} barb: {ST_BARBE_COORD} distance maxwell-barb={dist} m" )
    plot_place(place,tx)
    
    return place,tx,geometry


def create_dummy_place():
    geometry="dummy"
    #add ground and buildings
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [70, 24, 0]]))
    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, 0], [17, 24, 0]]))
    building_1 = geom.Building.by_polygon_and_height(square_1, 10,1)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, 0], [37, 2, 0]]))
    building_2 = geom.Building.by_polygon_and_height(square_2, 10,2)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, 0], [57, 24, 0]]))
    building_3 = geom.Building.by_polygon_and_height(square_3, 10,3)
    #add properties
    ground.properties=set_properties("ground")
    buildings=[building_1, building_2, building_3]
    for building in buildings:
        for polygon in building.polygons:
            polygon.properties=set_properties("appartments")
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground),buildings) 
    #add TX and RX
    tx = np.array([5., 12., 5.]).reshape(-1, 3)
    rx = np.array([65., 12., 5.]).reshape(-1, 3)
    place.add_set_of_points(rx)
    #save and plot
    place.to_json(filename="../data/dummy.json")
    plot_place(place,tx)
    return place, tx, geometry


def create_two_rays_place(npoints=20,plot=False):
    geometry="two_rays"
    #add ground
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [70, 24, 0]]))
    #add properties
    ground.properties=set_properties("ground")
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground))
    #add TX and RX
    tx = np.array([5., 12., 15.]).reshape(-1, 3)
    step=50
    rx0 = place.get_centroid().reshape(-1,3)+[0,0,5]
    for receiver in range(npoints):
        rx =rx0+np.array([receiver*step,0,0])
        place.add_set_of_points(rx)
    #save and plot
    place.to_json(filename="../data/two_rays.json")    
    if plot:
        plot_place(place,tx)
    return place, tx, geometry

#TODO: add receivers randomly otherwise they may end up inside buildings
def create_my_geometry():
    geometry="mygeometry"
    #create place
    geometry_filename="../data/TutorialTest/sciences.geojson"
    geometry_filename=geom.sample_geojson(geometry_filename,nBuildings=10)
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    # add TX and RX
    ground_center = place.get_centroid()
    tx = ground_center + [-50, 5, 1]
    rx = ground_center + [-100,10,1]
    tx = tx.reshape(-1, 3)
    rx = rx.reshape(-1, 3)
    place.add_set_of_points(rx)
    #plot
    plot_place(place,tx)
    return place,tx, geometry

if __name__ == '__main__':
    create_levant_place(npoints=10)
    
    
    

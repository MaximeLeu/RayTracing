#pylint: disable=invalid-name,line-too-long
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:52:10 2022

File containing the functions used to create the places.
@author: max
"""
#packages
import numpy as np
import matplotlib.pyplot as plt

#self written imports
import raytracing.geometry as geom
import plot_utils
from materials_properties import set_properties, LAMBDA




def plot_place(place,tx,show_normals=False):
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")
    place.center_3d_plot(ax)
    if show_normals:
        ax = place.plot3d(ax=ax,ret=True, poly_kwargs=dict(orientation=True, normal=True))
    else:
        ax = place.plot3d(ax=ax)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)
   # plt.savefig(f"../plots/thePlace.png", format='png', dpi=1000,bbox_inches='tight')
    return

def create_small_place(npoints=3):
    geometry="small"
    #create place
    geometry_filename='../data/small.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    
    #extend the ground
    # oriented_polyhedrons=place.polyhedra
    # ground = geom.Square.by_2_corner_points(np.array([[-50, 50, 0], [50, -200, 0]]))
    # ground.properties=set_properties("ground")
    # place = geom.OrientedPlace(geom.OrientedSurface(ground),oriented_polyhedrons)
    
    #add TX and RX
    tx = np.array([3, 38, 18]).reshape(-1, 3)
    rx=np.array([5,15,1.5]).reshape(-1,3)
    for _ in range(npoints):
        place.add_set_of_points(rx)
        rx =rx+np.array([0,-4,0])
    #save and plot
    place.to_json(filename="../data/small.json")
    plot_place(place,tx,show_normals=True)
    return place,tx,geometry


def create_levant_place(npoints=15):
    geometry="levant"
    #create place
    geometry_filename='../data/place_levant.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    #constants required for TX and RX positions
    ST_BARBE_COORD=np.array([-15,-46,0])
    ST_BARBE_COORD=ST_BARBE_COORD+np.array([10,0,0]) #offset to match claude's measures end point
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
    print(f"MAXWELL: {MAXWELL_COORDINATES} barb: {ST_BARBE_COORD} distance maxwell-barb={dist:.2f} m" )
    plot_place(place,tx,show_normals=True)

    return place,tx,geometry


def create_dummy_place():
    geometry="dummy"
    #add ground and buildings
    init_elevation=10
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, init_elevation], [70, 24, init_elevation]]))
    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, init_elevation], [17, 24, init_elevation]]))
    building_1 = geom.Building.by_polygon_and_height(square_1, 10,1)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, init_elevation], [37, 2, init_elevation]]))
    building_2 = geom.Building.by_polygon_and_height(square_2, 10,2)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, init_elevation], [57, 24, init_elevation]]))
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
    tx = np.array([5., 12., init_elevation+5.]).reshape(-1, 3)
    rx = np.array([65., 12.,init_elevation+ 5.]).reshape(-1, 3)
    place.add_set_of_points(rx)
    #save and plot
    place.to_json(filename=f"../data/{geometry}.json")
    plot_place(place,tx,show_normals=True)
    return place, tx, geometry


def create_two_rays_place(npoints=20,plot=False):
    geometry="two_rays"
    #add ground
    step=10#10*LAMBDA
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [step*npoints+50, 24, 0]]))
    #add properties
    ground.properties=set_properties("ground")
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground))
    #add TX and RX
    tx = np.array([5., 12., 30.]).reshape(-1, 3)
    rx0 = tx+np.array([50,0,-20])
    for receiver in range(npoints):
        rx =rx0+np.array([receiver*step,0,0])
        place.add_set_of_points(rx)
    #save and plot
    place.to_json(filename=f"../data/{geometry}.json")
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

    

def create_slanted_place(alpha):
    geometry="slanted"
    #add ground and buildings
    init_elevation=10
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, init_elevation], [70, 24, init_elevation]]))
    ground=ground.rotate(axis=np.array([0, 1, 0]), angle=alpha)
    ground=ground.rotate(axis=np.array([1, 0, 0]), angle=180) #othewise the normal is wrongly oriented

    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, init_elevation], [17, 24, init_elevation]]))
    building_1 = geom.Building.building_on_slope(square_1, ground,10)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, init_elevation], [37, 2, init_elevation]]))
    building_2 = geom.Building.building_on_slope(square_2, ground,10)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, init_elevation], [57, 24, init_elevation]]))
    building_3 = geom.Building.building_on_slope(square_3, ground,10)
    
    #add properties
    ground.properties=set_properties("ground")
    buildings=[building_1, building_2, building_3]
    for building in buildings:
        for polygon in building.polygons:
            polygon.properties=set_properties("appartments")
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground),buildings)
    #add TX and RX
    tx = np.array([5., 12., init_elevation+10]).reshape(-1, 3)
    rx = np.array([65., 12.,init_elevation+ 10.]).reshape(-1, 3)
    place.add_set_of_points(rx)
    #save and plot
    place.to_json(filename=f"../data/{geometry}.json")
    plot_place(place,tx,show_normals=True)
    return place, tx, geometry



if __name__ == '__main__':
    #create_levant_place(npoints=10)
    plt.close("all")
    #place,tx,geometry=create_two_rays_place(npoints=5)
    #plot_place(place, tx)
    place,tx,geometry=create_slanted_place(10)
    #place,tx,geometry=create_dummy_place()
    #place,tx,geometry=create_small_place(npoints=10)
    #place,tx,geometry=create_levant_place()
    
    




#pylint: disable=invalid-name,line-too-long
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:52:10 2022

File containing the functions used to create the places.
@author: Maxime Leurquin
"""
#packages
import numpy as np
import matplotlib.pyplot as plt

#self written imports
import raytracing.geometry as geom
import raytracing.plot_utils as plot_utils

from raytracing.materials_properties import set_properties, LAMBDA

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
    #plt.savefig(f"../plots/thePlace.png", format='png', dpi=1000,bbox_inches='tight')
    return

def create_small_place(npoints=3):
    geometry="small"
    #create place
    geometry_filename='../data/small.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
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
    #save
    place.to_json(filename=f"../data/{geometry}.json")
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
    #save
    place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry

def create_slanted_dummy(alpha):
    geometry="slanted_dummy"
    #add ground and buildings
    init_elevation=10
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, init_elevation], [70, 24, init_elevation]]))
    ground=ground.rotate(axis=np.array([0, 1, 0]), angle_deg=alpha)
    ground=ground.rotate(axis=np.array([1, 0, 0]), angle_deg=180) #othewise the normal is wrongly oriented

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
    #save
    place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry


def create_two_rays_place(npoints=20):
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
    #save
    place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry


#TODO: add receivers randomly otherwise they may end up inside buildings
def create_my_geometry():
    geometry="mygeometry"
    #create place
    geometry_filename="../data/TutorialTest/sciences.geojson"
    geometry_filename=geom.sample_geojson(geometry_filename,nBuildings=10)
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    # add TX and RX
    ground_center = place.get_centroid()
    tx = ground_center + [-50, 5, 1]
    rx = ground_center + [-100,10,1]
    tx = tx.reshape(-1, 3)
    rx = rx.reshape(-1, 3)
    place.add_set_of_points(rx)
    place.to_json(filename=f"../data/{geometry}.json")
    return place,tx, geometry

    


def create_flat_levant(npoints=15):
    geometry="levant"
    #create place
    geometry_filename='../data/place_levant.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    
    def define_levant_top_bottom():
        levant_top = np.array([57, -10, 0])
        levant_bottom = levant_top + np.array([-90, 0, 0])
        return levant_top, levant_bottom
    
    levant_top, levant_bottom=define_levant_top_bottom()
    
    def add_tx_rx(place, levant_top, levant_bottom, npoints):
        RX_HEIGHT = 1.2
        TX_HEIGHT = 1.2
        MAXWELL_HEIGHT=8.24
        tx = levant_top + np.array([2, 0, MAXWELL_HEIGHT + TX_HEIGHT])
        tx = tx.reshape(-1, 3)
        rx0 = levant_top + np.array([-30, 0, RX_HEIGHT])
        step = np.linalg.norm(levant_bottom - rx0) / npoints
        for receiver in range(npoints):
            rx = rx0 + np.array([-step * receiver, 0, 0])    
            place.add_set_of_points(rx.reshape(-1, 3)) 
        return place, tx
    
    def create_trees():
        #add trees
        tree_size=1
        tree_height=15
        tree1_spot=levant_top+np.array([-10,6,0])
        tree1=geom.Building.create_tree(tree1_spot,tree_size,tree_height,rotate=False)
        tree2_spot=tree1_spot+np.array([-15,0,0])
        tree2=geom.Building.create_tree(tree2_spot,tree_size,tree_height,rotate=False)
        tree3_spot=tree1_spot+np.array([0,-10,0])
        tree3=geom.Building.create_tree(tree3_spot,tree_size,tree_height,rotate=False)
        trees=[tree1,tree2,tree3]
        return trees
        
    trees=create_trees()
    for tree in trees:
        place.add_polyhedron(tree)
        
    place,tx=add_tx_rx(place, levant_top, levant_bottom, npoints)
    #save
    place.to_json(filename=f"../data/{geometry}.json")
    return place,tx,geometry


def create_slanted_levant(npoints=15):
    geometry="slanted_levant"
    geometry_filename='../data/place_levant.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    
    deltaH=3 #height difference between top and bottom of levant street
    barb=np.array([-140, -90, 0])
    vinci=np.array([-25,90,0])
    stevin=np.array([20,-90,deltaH])
    maxwell=np.array([140,90,deltaH])
    
    def create_grounds():
        ground1=geom.Square.by_2_corner_points(np.array([barb,vinci])) #flat ground between barb and vinci
        ground1=ground1.rotate(axis=np.array([0,1,0]), angle_deg=180)
        ground2=geom.Square.by_2_corner_points(np.array([vinci,stevin])) #slanted ground between vinci and stevin
        ground3=geom.Square.by_2_corner_points(np.array([stevin,maxwell])) #flat ground between stevin and maxwell
        ground3=ground3.rotate(axis=np.array([0,1,0]), angle_deg=180)
        ground1.properties=set_properties("road")
        ground2.properties=set_properties("road")
        ground3.properties=set_properties("road")
        grounds=[ground1,ground2,ground3]
        return grounds
    
    grounds=create_grounds()
    
    def define_levant_top_bottom(grounds):
        levant_top = np.array([57, -10, deltaH])
        points = [levant_top, levant_top + np.array([0, 0, 100])]
        levant_top = geom.polygon_line_intersection(grounds[2], points)
        levant_bottom = levant_top + (-90, 0, -deltaH)
        points = [levant_bottom, levant_bottom + np.array([0, 0, 100])]
        levant_bottom = geom.polygon_line_intersection(grounds[0], points)
        return levant_top, levant_bottom
    
    levant_top, levant_bottom=define_levant_top_bottom(grounds)
    
    def rebuild_buildings_on_slanted_grounds(place, grounds):
        buildings = []
        for i, polyhedron in enumerate(place.polyhedra):
            top = polyhedron.get_top_face()
            height = top.points[0][2]
            rebuilded = geom.Building.rebuild_building(top, grounds, height)
            for building in rebuilded:
                for rebuilt_polygon in building.polygons:
                    rebuilt_polygon.building_type = polyhedron.building_type
                    rebuilt_polygon.properties = set_properties(rebuilt_polygon.building_type)
                buildings.append(building)      
        return buildings
    
    buildings=rebuild_buildings_on_slanted_grounds(place,grounds)
    
    def create_trees():
        #add trees
        tree_size=1
        tree_height=15
        tree1_spot=levant_top+np.array([-10,6,0])
        tree1=geom.Building.create_tree(tree1_spot,tree_size,tree_height,rotate=False)
        tree2_spot=tree1_spot+np.array([-15,0,0])
        tree2=geom.Building.create_tree(tree2_spot,tree_size,tree_height,rotate=False)
        tree3_spot=tree1_spot+np.array([0,-10,0])
        tree3=geom.Building.create_tree(tree3_spot,tree_size,tree_height,rotate=False)
        trees=[tree1,tree2,tree3]
        return trees
        
    trees=create_trees()
    buildings.extend(trees)
    
    #rebuild the place
    place=geom.OrientedPlace(geom.OrientedSurface(grounds),buildings)
    #place.surface=place.surface.translate(np.array([0,0,100])) #to check if normals are well set

    def add_tx_rx(place, levant_top, levant_bottom, npoints):
        RX_HEIGHT = 1.2
        TX_HEIGHT = 1.2
        MAXWELL_HEIGHT=8.24
        tx = levant_top + np.array([2, 0, MAXWELL_HEIGHT + TX_HEIGHT])
        tx = tx.reshape(-1, 3)
        rx0 = levant_top + np.array([-30, 0, RX_HEIGHT])
        step = np.linalg.norm(levant_bottom - rx0) / npoints
        for receiver in range(npoints):
            rx = rx0 + np.array([-step * receiver, 0, 0])    
            if rx[0] < stevin[0]:  # in the slope
                rx = geom.polygon_line_intersection(grounds[1], [rx, rx + np.array([0, 0, 100])])
                rx = rx + np.array([0, 0, RX_HEIGHT])     
            if rx[0] < vinci[0]:  # on flat ground again
                rx[2] = RX_HEIGHT     
            place.add_set_of_points(rx.reshape(-1, 3)) 
        return place, tx

    place, tx=add_tx_rx(place, levant_top, levant_bottom, npoints)
    place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry

    
def test_place():
    geometry="test_place"
    # ground1 = geom.Square.by_2_corner_points(np.array([[0, 0, 10], [70, 24, 20]]))
    # ground1.properties=set_properties("ground")
    # ground2 = geom.Square.by_2_corner_points(np.array([[70, 24, 20], [90, 0, 20]]))
    # ground2.properties=set_properties("ground")
    # grounds=[ground1,ground2]
    
    barb=np.array([-100, -90, 0])
    vinci=np.array([-25,90,0])
    stevin=np.array([20,-90,10])
    maxwell=np.array([125,90,10])
    ground1=geom.Square.by_2_corner_points(np.array([barb,vinci])) #flat ground between stevin and maxwell
    ground1=ground1.rotate(axis=np.array([0,1,0]), angle_deg=180)
    ground2=geom.Square.by_2_corner_points(np.array([vinci,stevin])) #slanted ground between vinci and stevin
    ground3=geom.Square.by_2_corner_points(np.array([stevin,maxwell])) #flat ground between stevin and maxwell
    ground3=ground3.rotate(axis=np.array([0,1,0]), angle_deg=180)
    ground1.properties=set_properties("ground")
    ground2.properties=set_properties("ground")
    ground3.properties=set_properties("ground")
    grounds=[ground1,ground2,ground3]

    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, 5], [30, 24, 5]]))
    building_1 = geom.Building.building_on_slope(square_1, ground1,10)
    buildings=[building_1]
    for building in buildings:
        for polygon in building.polygons:
            polygon.properties=set_properties("appartments")
    #create place
    place = geom.OrientedPlace(grounds,buildings)
    tx = np.array([5., 12., 20]).reshape(-1, 3)
    place.to_json(filename=f"../data/{geometry}.json")
    plot_place(place,tx)
    #split the building
    top=place.polyhedra[0].get_top_face()
    
    splits=top.split([ground2,ground3])
    buildings=[]
    buildings.extend(geom.Building.rebuild_building(top,grounds,20))
    
    # for split in splits:
    #     buildings.append(geom.Building.by_polygon_and_height(split, 20, make_ccw=True, keep_ground=True,flat_roof=True))
        
    place = geom.OrientedPlace(grounds,buildings)   
    return place, tx, geometry
    


if __name__ == '__main__':
    plt.close("all")
    #place,tx,geometry=create_two_rays_place(npoints=5)
    #place,tx,geometry=create_slanted_dummy(alpha=10)
    #place,tx,geometry=create_dummy_place()
    #place,tx,geometry=create_small_place(npoints=10)
    #place,tx,geometry=create_flat_levant()
    #place,tx,geometry=create_slanted_levant(npoints=8)
    #place,tx,geometry=test_place()
    
    #TODO: WHY CANT CREATE TWO PLACES CONSECUTIVELY?
    
    #place,tx,geometry=create_dummy_place()
    
    place,tx,geometry=create_small_place(npoints=10)
    place,tx,geometry=create_small_place(npoints=10)
    #place,tx,geometry=create_flat_levant()
    
    plot_place(place, tx)
    
    




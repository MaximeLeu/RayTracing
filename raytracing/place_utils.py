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
import geopandas as gpd

#self written imports
import raytracing.geometry as geom
from raytracing.materials_properties import set_properties, LAMBDA


import raytracing.plot_utils as plot_utils
import raytracing.file_utils as file_utils
file_utils.chdir_to_file_dir(__file__)

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
    #plt.savefig(f"../results/plots/thePlace.png", format='png', dpi=1000,bbox_inches='tight')
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
    ground=ground.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, init_elevation], [17, 24, init_elevation]]))
    building_1 = geom.Building.by_polygon_and_height(polygon=square_1, height=10)
    square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, init_elevation], [37, 2, init_elevation]]))
    building_2 = geom.Building.by_polygon_and_height(polygon=square_2, height=10)
    square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, init_elevation], [57, 24, init_elevation]]))
    building_3 = geom.Building.by_polygon_and_height(polygon=square_3, height=10)
    #add properties
    ground.properties=set_properties("ground")
    buildings=[building_1, building_2, building_3]
    for building in buildings:
        building.building_type="appartments"
        building.apply_properties_to_polygons()
    #create place
    place = geom.OrientedPlace(geom.OrientedSurface(ground),buildings)
    #add TX and RX
    tx = np.array([5., 12., init_elevation+5.]).reshape(-1, 3)
    rx = np.array([65., 12.,init_elevation+ 5.]).reshape(-1, 3)
    place.add_set_of_points(rx)
    #save
    place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry



def create_two_rays_place(npoints=20):
    geometry="two_rays"
    #add ground
    step=LAMBDA#10#10*LAMBDA
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [step*npoints+50, 24, 0]]))
    ground=ground.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    ground.properties=set_properties("ground")
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


def find_crucial_coordinates(filename="../data/place_levant.geojson"):
    gdf = gpd.read_file(filename)
    gdf.to_crs(epsg=3035, inplace=True)
    geom.center_gdf(gdf)

    maxwell_point=gdf.loc[gdf['id']=="node/2368705254"]["geometry"].values[0]
    maxwell_entrance=np.array([maxwell_point.x,maxwell_point.y,0])
    
    levant_bottom=maxwell_entrance+np.array([-94,0,0])
    
    stevin_entrance_point=gdf.loc[gdf['id']=="node/8804430834"]["geometry"].values[0]
    stevin_entrance=np.array([stevin_entrance_point.x,stevin_entrance_point.y,0])
    
    reaumur_entrance_point=gdf.loc[gdf['id']=="node/2034353808"]["geometry"].values[0]
    reaumur_entrance=np.array([reaumur_entrance_point.x,reaumur_entrance_point.y,0])

    vinci_tree_point=gdf.loc[gdf['id']=="node/5710653390"]["geometry"].values[0]
    vinci_tree_spot=np.array([vinci_tree_point.x,vinci_tree_point.y,0])
    maxwell_tree_point=gdf.loc[gdf['id']=="node/5710653391"]["geometry"].values[0]
    maxwell_tree_spot=np.array([maxwell_tree_point.x,maxwell_tree_point.y,0])
    stevin_tree_point=gdf.loc[gdf['id']=="node/5710653392"]["geometry"].values[0]
    stevin_tree_spot=np.array([stevin_tree_point.x,stevin_tree_point.y,0])
    
    entrances=[maxwell_entrance,stevin_entrance,reaumur_entrance]
    tree_spots=[vinci_tree_spot,maxwell_tree_spot,stevin_tree_spot]
    bound_points=geom.get_bounds(gdf)
    return entrances,levant_bottom,bound_points,tree_spots
    


def create_flat_levant(npoints=15):
    geometry="flat_levant"
    #create place
    geometry_filename='../data/place_levant.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    
    entrances,levant_bottom,bound_points,tree_spots=find_crucial_coordinates(geometry_filename)
    maxwell_entrance,stevin_entrance,reaumur_entrance=entrances
    vinci_tree_spot,maxwell_tree_spot,stevin_tree_spot=tree_spots
    
    def add_tx_rx(place, maxwell, levant_bottom, npoints):
        RX_HEIGHT = 1.2
        TX_HEIGHT = 1.2
        MAXWELL_HEIGHT=8.24
        tx = maxwell + np.array([1, 0, MAXWELL_HEIGHT + TX_HEIGHT])
        tx = tx.reshape(-1, 3)
        rx0 = maxwell + np.array([-34, 0, RX_HEIGHT])
        step = np.linalg.norm(levant_bottom - rx0) / npoints
        for receiver in range(npoints):
            rx = rx0 + np.array([-step * receiver, 0, 0])    
            place.add_set_of_points(rx.reshape(-1, 3)) 
        return tx
    
    def add_trees(place):
        #add trees
        tree_size=1
        tree_height=15
        vinci_tree=geom.Building.create_tree(vinci_tree_spot,tree_size,tree_height,rotate=False)
        maxwell_tree=geom.Building.create_tree(maxwell_tree_spot,tree_size,tree_height,rotate=False)
        stevin_tree=geom.Building.create_tree(stevin_tree_spot,tree_size,tree_height,rotate=False)
        trees=[vinci_tree,maxwell_tree,stevin_tree]
        for tree in trees:
            place.add_polyhedron(tree)
        return trees
    
    add_trees(place)
    tx=add_tx_rx(place, maxwell_entrance, levant_bottom, npoints)
    place.to_json(filename=f"../data/{geometry}.json")
    return place,tx,geometry


def create_slanted_levant(npoints=15):
    geometry="slanted_levant"
    geometry_filename='../data/place_levant.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    
    entrances,levant_bottom,bound_points,tree_spots=find_crucial_coordinates(geometry_filename)
    maxwell_entrance,stevin_entrance,reaumur_entrance=entrances
    
    deltaH=3 #height difference between top and bottom of levant street
    bottom_left=bound_points[0] #behind the barb
    top_right=bound_points[2]+np.array([0,0,deltaH])#behind the maxwell
    low_right=np.array([reaumur_entrance[0],top_right[1],0]) #start of slope
    high_left=np.array([stevin_entrance[0],bottom_left[1],deltaH]) #end of slope
    
    def create_grounds(bottom_left,top_right,low_right,high_left):
        ground1=geom.Square.by_2_corner_points(np.array([bottom_left,low_right])) #flat ground between barb and vinci
        ground1=ground1.rotate(axis=np.array([0,1,0]), angle_deg=180)
        ground2=geom.Square.by_2_corner_points(np.array([low_right,high_left])) #slanted ground between vinci and stevin
        ground3=geom.Square.by_2_corner_points(np.array([high_left,top_right])) #flat ground between stevin and maxwell
        ground3=ground3.rotate(axis=np.array([0,1,0]), angle_deg=180)
        ground1.properties=set_properties("road")
        ground2.properties=set_properties("road")
        ground3.properties=set_properties("road")
        grounds=[ground1,ground2,ground3]
        return grounds
    
    grounds=create_grounds(bottom_left,top_right,low_right,high_left)
    
    def define_levant_top_bottom(grounds,maxwell_entrance,levant_bottom):
        points = [maxwell_entrance, maxwell_entrance + np.array([0, 0, 100])]
        maxwell_entrance = geom.polygon_line_intersection(grounds[2], points)
        
        points = [levant_bottom, levant_bottom + np.array([0, 0, 100])]
        levant_bottom = geom.polygon_line_intersection(grounds[0], points)
        return maxwell_entrance, levant_bottom
    
    maxwell_entrance, levant_bottom=define_levant_top_bottom(grounds,maxwell_entrance,levant_bottom)
    
    def rebuild_buildings_on_slanted_grounds(place, grounds):
        buildings = []
        for i, polyhedron in enumerate(place.polyhedra):
            top = polyhedron.get_top_face()
            height = top.points[0][2]
            rebuilded = geom.Building.rebuild_building(top, grounds, height)
            for building in rebuilded: #the building maybe split on multiple grounds
                building.building_type=polyhedron.building_type
                building.apply_properties_to_polygons()
                buildings.append(building)      
        return buildings
    
    buildings=rebuild_buildings_on_slanted_grounds(place,grounds)
    
    def add_trees(tree_spots):
        vinci_tree_spot,maxwell_tree_spot,stevin_tree_spot=tree_spots
        vinci_tree_spot+=np.array([0,0,deltaH])
        maxwell_tree_spot+=np.array([0,0,deltaH])
        stevin_tree_spot+=np.array([0,0,deltaH])
        #add trees
        tree_size=1
        tree_height=15
        vinci_tree=geom.Building.create_tree(vinci_tree_spot,tree_size,tree_height,rotate=False)
        maxwell_tree=geom.Building.create_tree(maxwell_tree_spot,tree_size,tree_height,rotate=False)
        stevin_tree=geom.Building.create_tree(stevin_tree_spot,tree_size,tree_height,rotate=False)
        trees=[vinci_tree,maxwell_tree,stevin_tree]
        return trees
    trees=add_trees(tree_spots)
    buildings.extend(trees)
    
    #rebuild the place
    place=geom.OrientedPlace(geom.OrientedSurface(grounds),buildings)
    #place.surface=place.surface.translate(np.array([0,0,100])) #to check if ground normals are well set

    def add_tx_rx(place, maxwell_entrance, levant_bottom, npoints):
        RX_HEIGHT = 1.2
        TX_HEIGHT = 1.2
        MAXWELL_HEIGHT=8.24
        tx = maxwell_entrance + np.array([1, 0, MAXWELL_HEIGHT + TX_HEIGHT])
        tx = tx.reshape(-1, 3)
        rx0 = maxwell_entrance + np.array([-34, 0, RX_HEIGHT])
        step = np.linalg.norm(levant_bottom - rx0) / npoints
        for receiver in range(npoints):
            rx = rx0 + np.array([-step * receiver, 0, 0])    
            if rx[0] < high_left[0]:  # in the slope
                rx = geom.polygon_line_intersection(grounds[1], [rx, rx + np.array([0, 0, 100])])
                rx = rx + np.array([0, 0, RX_HEIGHT])     
            if rx[0] < low_right[0]:  # on flat ground again
                rx[2] = RX_HEIGHT     
            place.add_set_of_points(rx.reshape(-1, 3)) 
        return place, tx

    place, tx=add_tx_rx(place, maxwell_entrance, levant_bottom, npoints)
    place.to_json(filename=f"../data/{geometry}.json")
    
    return place, tx, geometry

    

    


if __name__ == '__main__':
    plt.close("all")
    #place,tx,geometry=create_small_place(npoints=10)
    #place,tx,geometry=create_flat_levant()
    
    place,tx,geometry=create_slanted_levant(npoints=8)
    #place,tx,geometry=test_place()
    #place,tx,geometry=create_two_rays_place(npoints=5)
    
    #place,tx,geometry=create_dummy_place()
    plot_place(place, tx,show_normals=True)
    
    
    
    
    




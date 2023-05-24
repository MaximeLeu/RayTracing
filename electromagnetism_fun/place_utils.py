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
import random
import json
from pathlib import Path
#self written imports
import raytracing.geometry as geom
from electromagnetism_fun.materials_properties import set_properties, LAMBDA
from scipy.spatial.transform import Rotation

from shapely.geometry import LineString, MultiPolygon, Polygon,Point
from shapely.ops import split
from scipy.spatial.distance import cdist

import shapely

import raytracing.plot_utils as plot_utils
import raytracing.file_utils as file_utils
file_utils.chdir_to_file_dir(__file__)

def plot_place(place,tx,show_normals=False,name=None):
    if name is None:
        name='the place'
    fig = plt.figure(name,dpi=150)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x',fontsize=15)
    ax.set_ylabel('y',fontsize=15)
    ax.set_zlabel('z',fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")
    place.center_3d_plot(ax)
    if show_normals:
        ax = place.plot3d(ax=ax,ret=True, poly_kwargs=dict(orientation=True, normal=True))
    else:
        ax = place.plot3d(ax=ax)
    plt.legend(fontsize=15)
    #plt.savefig(f"../results/plots/thePlace.png", format='png', dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.01)
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
    #place.to_json(filename=f"../data/{geometry}.json")
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
    ground.properties=set_properties("road")
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
    #place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry


def create_two_rays_place(npoints=20):
    geometry="two_rays"
    #add ground
    step=1#LAMBDA#10#10*LAMBDA
    ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [step*npoints+50, 24, 0]]))
    ground=ground.rotate(axis=np.array([0,1,0]), angle_deg=180)
    
    ground.properties=set_properties("road")
    place = geom.OrientedPlace(geom.OrientedSurface(ground))
    #add TX and RX
    tx = np.array([5., 12., 30.]).reshape(-1, 3)
    rx0 = tx+np.array([50,0,-20])
    for receiver in range(npoints):
        rx =rx0+np.array([receiver*step,0,0])
        place.add_set_of_points(rx)
    #save
    #place.to_json(filename=f"../data/{geometry}.json")
    return place, tx, geometry


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


class Place_du_levant():
    @staticmethod
    def levant_find_crucial_coordinates(filename="../data/place_levant_edited.geojson"):
        #Only useful to create the levant scenarios.
        gdf = gpd.read_file(filename)
        gdf.to_crs(epsg=3035, inplace=True)
        geom.center_gdf(gdf)

        maxwell_point=gdf.loc[gdf['id']=="node/2368705254"]["geometry"].values[0]
        maxwell_entrance=np.array([maxwell_point.x,maxwell_point.y,0])
        
        levant_bottom=maxwell_entrance+np.array([-94,0,0])#february measures
        #levant_bottom=maxwell_entrance+np.array([-83,0,0]) #october measures
        
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

    @staticmethod
    def levant_add_trees(tree_spots):
        rotate = False
        vinci_tree_spot, maxwell_tree_spot, stevin_tree_spot = tree_spots
        
        trunk_size,trunk_height=[0.5,4]
        crown_size, crown_height = [2, 5]
        vinci_tree_trunk = geom.Building.create_tree_trunk(vinci_tree_spot, trunk_size, trunk_height,rotate,\
                                                           crown_size, crown_height)                                                                                                    
        vinci_tree_crown = geom.Building.create_tree_crown(vinci_tree_trunk, crown_size, crown_height,rotate)
            
        trunk_size,trunk_height=[0.5,4.5]
        crown_size, crown_height = [2, 4.5]
        maxwell_tree_trunk = geom.Building.create_tree_trunk(maxwell_tree_spot, trunk_size, trunk_height,rotate,\
                                                             crown_size, crown_height)
        maxwell_tree_crown = geom.Building.create_tree_crown(maxwell_tree_trunk, crown_size, crown_height,rotate)
        
        trunk_size,trunk_height=[0.5,6.5]
        crown_size, crown_height = [2, 2]
        stevin_tree_trunk = geom.Building.create_tree_trunk(stevin_tree_spot, trunk_size, trunk_height,rotate,\
                                                            crown_size, crown_height,)
        stevin_tree_crown = geom.Building.create_tree_crown(stevin_tree_trunk, crown_size, crown_height,rotate)
        
        tree_trunks = [vinci_tree_trunk, maxwell_tree_trunk, stevin_tree_trunk]
        tree_crowns = [vinci_tree_crown, maxwell_tree_crown, stevin_tree_crown]
        return tree_trunks, tree_crowns
   
    @staticmethod
    def create_flat_levant(npoints=15):
        geometry="flat_levant"
        geometry_filename='../data/place_levant_edited.geojson'
        preprocessed_name=geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(preprocessed_name)
        
        entrances,levant_bottom,_,tree_spots=Place_du_levant.levant_find_crucial_coordinates(geometry_filename)
        maxwell_entrance,_,_=entrances
        
        def add_tx_rx(place, maxwell, levant_bottom, npoints):
            RX_HEIGHT = 1.2
            TX_HEIGHT = 1.2
            MAXWELL_HEIGHT=17.5
            tx=maxwell_entrance+np.array([23,6,MAXWELL_HEIGHT+TX_HEIGHT])
            tx = tx.reshape(-1, 3)
            rx0 = maxwell + np.array([-34, 0, RX_HEIGHT])
            step = np.linalg.norm(levant_bottom - rx0) / npoints
            for receiver in range(npoints):
                rx = rx0 + np.array([-step * receiver, 0, 0])    
                place.add_set_of_points(rx.reshape(-1, 3)) 
            return tx    

        #construct stairs
        # stairs_polyhedron=place.polyhedra[3]
        # sh_rectangle=stairs_polyhedron.get_top_face()
        # polygons=split_rectangle(sh_rectangle,nx=1,ny=1)  
        # polyhedrons=build_staircase(polygons,place.surface.polygons[0])
        # del place.polyhedra[3]
        # place.polyhedra.extend(polyhedrons)

        
        tree_trunks,tree_crowns=Place_du_levant.levant_add_trees(tree_spots)
        place.polyhedra.extend(tree_trunks)
        #place.polyhedra.extend(tree_crowns) #when added here tree crowns are blocking rays.

        tx=add_tx_rx(place, maxwell_entrance, levant_bottom, npoints)
        #place.to_json(filename=f"../data/{geometry}.json")
        return place,tx,geometry

    @staticmethod
    def create_slanted_levant(npoints=15):
        
        geometry="slanted_levant"
        geometry_filename='../data/place_levant_edited.geojson'
        preprocessed_name=geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(preprocessed_name)
        
        entrances,levant_bottom,bound_points,tree_spots=Place_du_levant.levant_find_crucial_coordinates(geometry_filename)
        maxwell_entrance,stevin_entrance,reaumur_entrance=entrances
        
        deltaH=3 #height difference between top and bottom of levant street
        tree_spots = [arr + np.array([0,0,deltaH]) for arr in tree_spots]
        
        bottom_left=bound_points[0] #behind the barb
        top_right=bound_points[2]+np.array([0,0,deltaH])#behind the maxwell
        low_right=np.array([reaumur_entrance[0],top_right[1],0]) #start of slope
        high_left=np.array([stevin_entrance[0],bottom_left[1],deltaH]) #end of slope
        
        def create_grounds(bottom_left,top_right,low_right,high_left):
            ground1=geom.Square.by_2_corner_points(np.array([bottom_left,low_right])) #flat ground between barb and vinci
            ground2=geom.Square.by_2_corner_points(np.array([low_right,high_left])) #slanted ground between vinci and stevin
            ground3=geom.Square.by_2_corner_points(np.array([high_left,top_right])) #flat ground between stevin and maxwell
            
            #ensure normals are pointing upwards
            ground1=ground1.rotate(axis=np.array([0,1,0]), angle_deg=180)
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
        
        buildings=rebuild_buildings_on_slanted_grounds(place,grounds)
        tree_trunks,tree_crowns=Place_du_levant.levant_add_trees(tree_spots)
        buildings.extend(tree_trunks)
        #buildings.extend(tree_crowns) #when added here tree crowns are blocking rays.
        
        #rebuild the place
        place=geom.OrientedPlace(geom.OrientedSurface(grounds),buildings)
        #place.surface=place.surface.translate(np.array([0,0,100])) #to check easily if ground normals are well set

        def add_tx_rx(place, maxwell_entrance, levant_bottom, npoints):
            RX_HEIGHT = 1.2
            TX_HEIGHT = 1.2
            MAXWELL_HEIGHT=17.5
            tx=maxwell_entrance+np.array([23,6,MAXWELL_HEIGHT+TX_HEIGHT])
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
        #place.to_json(filename=f"../data/{geometry}.json")
        
        #build stairs
        # stairs_polyhedron=place.polyhedra[4]
        # fig = plt.figure("the place",dpi=300)
        # ax = fig.add_subplot(projection='3d')
        # ax=stairs_polyhedron.plot3d(ax=ax)
        # plt.show()
        # sh_rectangle=stairs_polyhedron.get_top_face()
        # polygons=split_rectangle(sh_rectangle,nx=8,ny=1)  
        # polyhedrons=build_staircase(polygons,place.surface.polygons[1])
        # del place.polyhedra[4]
        # place.polyhedra.extend(polyhedrons)
        
        print(f"MAXWELL_ENTRANCE: {maxwell_entrance}")
        return place, tx, geometry

    @staticmethod
    def split_rectangle(polygon,nx,ny):
        #given an OrientedPolygon rectangle split it in nx rows and ny columns. Returns a list of shapely polygons.
        #code is copied from:
        #https://stackoverflow.com/questions/58283684/how-to-divide-a-rectangle-in-specific-number-of-rows-and-columns
        
        polygon=polygon.get_shapely()
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny, maxx, maxy = polygon.bounds
        dx = (maxx - minx) / nx  # width of a small part
        dy = (maxy - miny) / ny  # height of a small part
        horizontal_splitters = [LineString([(minx, miny + i*dy), (maxx, miny + i*dy)]) for i in range(ny)]
        vertical_splitters = [LineString([(minx + i*dx, miny), (minx + i*dx, maxy)]) for i in range(nx)]
        splitters = horizontal_splitters + vertical_splitters
        result = polygon
        for splitter in splitters:
            result = MultiPolygon(split(result, splitter))
        polygons = [geom for geom in result.geoms]
        
        plot=False
        if plot:
            fig = plt.figure("splitted rectangles")
            fig.set_dpi(300)
            ax = fig.add_subplot(projection='3d')
            ax=plot_utils.ensure_axis_orthonormal(ax)
            for polygon in polygons:
                polygon=geom.OrientedPolygon.shapely_to_oriented_polygon(polygon)
                polygon.plot3d(facecolor=(0,1,1,1),ax=ax,alpha=1,normal=False)      
            fig.show()     
        return polygons
          
    @staticmethod
    def build_staircase(polygons,ground):
        #given list of polygons, for each polygon build a polyhedron on the ground, and one a few meters above.
        roof_height=3
        polyhedrons=[]
        for polygon in polygons:
            polygon=geom.OrientedPolygon.shapely_to_oriented_polygon(polygon)
            bottom_polyhedron=geom.Building.building_on_slope(polygon,ground,height=0.5,id=None)
            
            bottom_polyhedron_z=bottom_polyhedron.get_top_face().points[0][2]      
            polygon=polygon.translate(np.array([0,0,bottom_polyhedron_z+roof_height]))
            top_polyhedron=geom.Building.by_polygon_and_height(polygon, height= 0.5, id=None,make_ccw=True, keep_ground=True,flat_roof=True)
            
            bottom_polyhedron.building_type="road"
            top_polyhedron.building_type="road"
            bottom_polyhedron.apply_properties_to_polygons()
            top_polyhedron.apply_properties_to_polygons()
            
            polyhedrons.extend([bottom_polyhedron,top_polyhedron])
        
        plot=False
        if plot:
            fig = plt.figure("splitted rectangles")
            fig.set_dpi(300)
            ax = fig.add_subplot(projection='3d')
            for polyhedron in polyhedrons:
                polyhedron.plot3d(facecolor=(0,1,1,1),ax=ax,alpha=1,normal=False)
            fig.show()
            
        return polyhedrons




class Place_saint_jean():
    @staticmethod
    def gen_place_saint_jean():
        geometry_filename='../data/place_saint_jean/place_saint_jean.geojson'
        preprocessed_name=geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(preprocessed_name)
        return place
    
    @staticmethod
    def create_place_saint_jean(points_filename,tx):
        geometry="place_saint_jean"
        place = Place_saint_jean.gen_place_saint_jean()
        points=Place_saint_jean.load_points_from_file(points_filename)
        place.add_set_of_points(points)
        
        if tx is None:
            #tx = np.array([25, 2.5, 18])#in the intersection, fails ....
            tx=np.array([72,40,2])#near big mama
            #tx=np.array([-93,80,2])#near conservatoire
        
        #place.surface=place.surface.translate(np.array([0,0,100])) #to check easily if ground normals are well set
        tx=tx.reshape(-1,3)
        #place.to_json(filename=f"../data/place_saint_jean/{geometry}.json")
        return place,tx,geometry
    
    @staticmethod
    def create_slanted_place_saint_jean(points_filename,tx,tx_on_ground=True):
        geometry="slanted_place_saint_jean"
        place = Place_saint_jean.gen_place_saint_jean()
        points=Place_saint_jean.load_points_from_file(points_filename)
        place.add_set_of_points(points)
        
        RX_HEIGHT=1.2
        TX_HEIGHT=3
        
        def create_grounds():
            bottom_left=np.array([-170,-150,16])
            top_right=np.array([150,150,34])
            ground1=geom.Square.by_2_corner_points(np.array([bottom_left,top_right]))
            #ensure normals are pointing upwards
            ground1=ground1.rotate(axis=np.array([0,1,0]), angle_deg=180) 
            ground1=ground1.rotate(axis=np.array([0,0,1]), angle_deg=-80)
            ground1.properties=set_properties("road")
            ground1.part='ground'
            grounds=[ground1]
            return grounds
            
        grounds=create_grounds()
        buildings=rebuild_buildings_on_slanted_grounds(place,grounds)  
        #rebuild the place
        place=geom.OrientedPlace(geom.OrientedSurface(grounds),buildings)
        #place.surface=place.surface.translate(np.array([0,0,100])) #to check easily if ground normals are well set 
        
        def add_rx(place,points_filename):
            points=Place_saint_jean.load_points_from_file(points_filename)
            print(f"loaded {len(points)} points from file")
            grounds=place.surface.polygons
            for i,point in enumerate(points):
                shPoint = Point(point[0], point[1])
                #find above which ground the point is
                for ground in grounds:
                    if ground.get_shapely().contains(shPoint):
                        #get intersection between point and ground
                        #then put the point ..m away from the ground
                        line=[point,point+np.array([0,0,1])]
                        intersection=geom.polygon_line_intersection(ground,line)
                        points[i]=intersection+np.array([0,0,RX_HEIGHT])
            place.add_set_of_points(points)
            print(f"added {len(points)} points to the place.")
        add_rx(place,points_filename)
        
        def add_tx(tx,base_surface):
            #adds the tx TX_HEIGHT away from the base_surface.
            if tx is None:
                tx = np.array([25, 2, 18])#in the intersection
            line=[tx,tx+np.array([0,0,1])]
            intersection=geom.polygon_line_intersection(base_surface,line)
            tx=intersection+np.array([0,0,TX_HEIGHT])
            tx = tx.reshape(-1, 3)
            return tx
        
        if tx_on_ground is True:
            ground=place.surface.polygons[0]
            tx=add_tx(tx,ground)
        else:
            tx=add_tx(tx,place.polyhedra[29].get_top_face()) #31 centraal=place.polyhedra[29]
        
        #place.to_json(filename=f"../data/place_saint_jean/{geometry}.json")
        return place,tx,geometry
    
    @staticmethod
    def generate_spaced_points_in_polygon(polygon: Polygon, n: int, d: float):
        #takes a shapely polygon as argument
        #randomly generates points inside the polygon that at least have a distance d between them.
        #can run infinitely if d is too large compared to the number of points asked
        points = []
        minx, miny, maxx, maxy = polygon.bounds
        while len(points) < n:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            point = Point(x, y)
            if polygon.contains(point) and point.distance(polygon.boundary) > 1:
                if len(points) == 0:
                    points.append(point)
                else:
                    coords = np.array([p.coords[0] for p in points])
                    distances = cdist(coords, np.array([(x, y)]))
                    if np.min(distances) >= d:
                        points.append(point)
        return points
    
    @staticmethod
    def create_points(npoints=16*5,d=4):
        street_polygon=Place_saint_jean.select_saint_jean_streets_hardcoded()
        points=Place_saint_jean.generate_spaced_points_in_polygon(street_polygon,npoints,d)
        #add 3th dimension
        RX_HEIGHT=1.2
        points = [(p.coords[0][0], p.coords[0][1], RX_HEIGHT) for p in points]
        filename=f"../data/place_saint_jean/place_saint_jean_{npoints}_points_d_{d}.json"
        with open(filename, 'w') as f:
            json.dump(points, f)
        print(f"generated and wrote points to {filename}")
        return points,filename
    
    @staticmethod
    def load_points_from_file(filename):
        with open(filename, 'r') as f:
            points = json.load(f)
        return [tuple(point) for point in points]
    
    @staticmethod
    def plot_points(filename):
        points=Place_saint_jean.load_points_from_file(filename)
        street_polygon=Place_saint_jean.select_saint_jean_streets_hardcoded()
        ax=plot_utils.plot_shapely(street_polygon)
        for point in points:
            ax=plot_utils.plot_shapely(shapely.geometry.Point(point),ax)
        return 
    
    @staticmethod
    def select_saint_jean_streets_hardcoded():
        place = Place_saint_jean.gen_place_saint_jean()
        
        zone = MultiPolygon()
        for polyhedron in place.polyhedra:
            top=polyhedron.get_top_face().get_shapely()
            zone=zone.union(top)
        
        zone=zone.simplify(tolerance=1e-2)
        
        conservatoire=zone.geoms[0]
        conservatoire_points=conservatoire.exterior.coords[8:19]
        conservatoire_points.extend(conservatoire.exterior.coords[21:34])
      
        natural_corner=zone.geoms[1]
        natural_corner_points=natural_corner.exterior.coords[20::]
        natural_corner_points.append(natural_corner.exterior.coords[1])
       
        studio_baxton=zone.geoms[2]
        studio_baxton_points=studio_baxton.exterior.coords[20:-6]
      
        parlement=zone.geoms[3]
        parlement_points=parlement.exterior.coords[7:57]
       
        big_mama=zone.geoms[4]
        big_mama_points=big_mama.exterior.coords[-10::]
        big_mama_points.extend(big_mama.exterior.coords[:6])
        
        centraal=zone.geoms[5]
        centraal_points=centraal.exterior.coords[8:11]
        
        vdab=zone.geoms[6]
        vdab_points=vdab.exterior.coords[3:5]
          
        under_baxton=zone.geoms[7]
        under_baxton_points=under_baxton.exterior.coords[11:14]
           
        manneken=zone.geoms[8]
        manneken_points=manneken.exterior.coords[2:7]
           
        street_points=parlement_points
        street_points.extend(conservatoire_points)
        street_points.extend(manneken_points)
        street_points.extend(studio_baxton_points)
        street_points.extend(under_baxton_points)
        street_points.extend(natural_corner_points)
        street_points.extend(vdab_points)
        street_points.extend(centraal_points)
        street_points.extend(big_mama_points)
        
        # ax=plot_utils.plot_shapely(zone)
        # for point in street_points:
        #     ax=plot_utils.plot_shapely(shapely.geometry.Point(point),ax)   
        
        street_polygon=shapely.geometry.Polygon(street_points)
        return street_polygon
    
    @staticmethod
    def set_heights(filename='../data/place_saint_jean/place_saint_jean.geojson'):
        #run once after having extracted the geojson from overpass turbo
        #heights data comes from google earth
        #overwrites the original geojson
        gdf = gpd.read_file(filename)
        gdf=gdf.copy()
        
        #left branch- rue du chene
        gdf.loc[gdf['id']=="way/132534306","height"] = 18
        gdf.loc[gdf['id']=="way/224649029","height"] = 17
        gdf.loc[gdf['id']=="way/224649043","height"] = 17.5
        gdf.loc[gdf['id']=="way/224649045","height"] = 18.3
        gdf.loc[gdf['id']=="way/224649048", "height"] = 15.7
        gdf.loc[gdf['id']=="way/224649008", "height"] = 15
        gdf.loc[gdf['id']=="way/224649010", "height"] = 16.2
        gdf.loc[gdf['id']=="way/224649013", "height"] = 17.5
        gdf.loc[gdf['id']=="way/224649016", "height"] = 18
        gdf.loc[gdf['id']=="way/224649017", "height"] = 15.4
        gdf.loc[gdf['id']=="way/224649020", "height"] = 17.9
        
        gdf.loc[gdf['id']=="way/257712523", "height"] = 25
        gdf.loc[gdf['id']=="way/257712322", "height"] = 25.3 
        gdf.loc[gdf['id']=="way/257712313", "height"] = 25.1
        gdf.loc[gdf['id']=="way/257712612", "height"] = 22
        gdf.loc[gdf['id']=="way/257712467", "height"] = 17
        gdf.loc[gdf['id']=="way/257712240", "height"] = 15
        gdf.loc[gdf['id']=="way/257712430", "height"] = 15
        
        # right branch -rue de lescalier
        gdf.loc[gdf['id']=="way/224649059", "height"] = 15.9
        gdf.loc[gdf['id']=="way/224649042", "height"] = 17
        gdf.loc[gdf['id']=="way/224649055", "height"] = 16.9
        gdf.loc[gdf['id']=="way/224649041", "height"] = 15.2
        gdf.loc[gdf['id']=="way/224649067", "height"] = 16.6
        gdf.loc[gdf['id']=="way/224649040", "height"] = 18
        gdf.loc[gdf['id']=="way/224649039", "height"] = 17.7
        gdf.loc[gdf['id']=="way/224649038", "height"] = 16.3
        gdf.loc[gdf['id']=="way/224649056", "height"] = 22 #big mama
        gdf.loc[gdf['id']=="way/224649036", "height"] = 18
        gdf.loc[gdf['id']=="way/224649007", "height"] = 17.5
        gdf.loc[gdf['id']=="way/224649034", "height"] = 16.9
        gdf.loc[gdf['id']=="way/224649069", "height"] = 17
        gdf.loc[gdf['id']=="way/144256881", "height"] = 16
        
        gdf.loc[gdf['id']=="way/472952353", "height"] = 24
        gdf.loc[gdf['id']=="way/224649053", "height"] = 48#48m centraal
        
        # rue de dinant
        gdf.loc[gdf['id']=="way/257712250", "height"] = 18
        gdf.loc[gdf['id']=="way/257712276", "height"] = 17
        gdf.loc[gdf['id']=="way/257712333", "height"] = 17.5
        
        #middle branch
        gdf.loc[gdf['id']=="way/257712524", "height"] = 21 #bx beerbox
        gdf.loc[gdf['id']=="way/257712412", "height"] = 14
        gdf.loc[gdf['id']=="way/257712444", "height"] = 20
        gdf.loc[gdf['id']=="way/257712251", "height"] = 21
        gdf.loc[gdf['id']=="relation/3459600", "height"] = 21 #conservatoire
        gdf.loc[gdf['id']=="way/257712542", "height"] = 18
        gdf.loc[gdf['id']=="way/257712325", "height"] = 18
        gdf.loc[gdf['id']=="way/257712489", "height"] = 18
        gdf.loc[gdf['id']=="way/257712345", "height"] = 18
        gdf.loc[gdf['id']=="way/1094964729", "height"] = 18
        gdf.loc[gdf['id']=="way/132394539", "height"] = 15
        gdf.loc[gdf['id']=="way/224649021", "height"] = 22
        gdf.loc[gdf['id']=="way/224649022", "height"] = 21
        gdf.loc[gdf['id']=="way/224649065", "height"] = 21.5
        gdf.loc[gdf['id']=="way/224649025", "height"] = 22
        gdf.loc[gdf['id']=="way/224649027", "height"] = 21.6
        gdf.loc[gdf['id']=="way/224649030", "height"] = 20.4
        gdf.loc[gdf['id']=="way/224649032", "height"] = 21.9
        gdf.loc[gdf['id']=="way/472952353", "height"] = 22.4
        gdf.loc[gdf['id']=="way/224649058", "height"] = 20
        gdf.loc[gdf['id']=="way/132394513", "height"] = 18#natural corner
        gdf.loc[gdf['id']=="way/224649054", "height"] = 20#vdab
               
        gdf.to_file(filename, driver="GeoJSON")
        return
    
     

if __name__ == '__main__':
    plt.close("all")
    Place_saint_jean.set_heights()
    
    #place,tx,geometry=create_small_place(npoints=10)
    #place,tx,geometry=Place_du_levant.create_flat_levant(npoints=16)
    #place,tx,geometry=Place_du_levant.create_slanted_levant(npoints=16*5)
    #place,tx,geometry=create_two_rays_place(npoints=5)
    
    #points,points_filename=Place_saint_jean.create_points(npoints=16,d=4)
    points_filename="../data/place_saint_jean/place_saint_jean_16_points_d_4.json"
    #Place_saint_jean.plot_points(points_filename)
    
    #place,tx,geometry=Place_saint_jean.create_place_saint_jean(points_filename=points_filename)
    tx_at_pis=np.array([-102,92,0])
    tx_at_pis+=np.array([+6,0,0])
    
    tx_at_escalier=np.array([61,-93,0])
    tx_at_escalier+=np.array([-8,0,0])
    
    tx_at_vieille_ble=np.array([73,40,0])
    tx_at_vieille_ble_side=tx_at_vieille_ble+np.array([-3,1.5,0])
    
    tx_in_middle=np.array([35,-6,0])
    
    tx_at_centraal=np.array([69,-54,0])
    
    place,tx,geometry=Place_saint_jean.create_slanted_place_saint_jean(points_filename=points_filename,tx=tx_at_centraal,tx_on_ground=False)
    #place.set_of_points=np.array([])
    
    plot_place(place, tx,show_normals=False)
    
    
    

   
    
   
      
    
    
    
    
    




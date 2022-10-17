# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:33:28 2022

@author: maxime
"""

from ray_tracing import RayTracingProblem
import raytracing.geometry as geom
import numpy as np
from raytracing import plot_utils
import matplotlib.pyplot as plt
from electromagnetism import my_field_computation
from raytracing import file_utils

#driver code
if __name__ == '__main__':
    plt.close('all') #close all previous plots
    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    geometry = 'small'  # You can change to use another geometry
    geometry_filename=""
    if geometry == 'small':
        geometry_filename='../data/small.geojson'
        
        geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(geometry_filename)
        
        # 2. Create TX and RX

        domain = place.get_domain()
        ground_center = place.get_centroid()

        tx = ground_center + [-50, 5, 1]
        rx = ground_center + np.array([
            [35, 5, 5],
            [35, -5, 5],
            [10, -3, -5]
        ])
        rx = rx[2, :]
        tx = tx.reshape(-1, 3)
        rx = rx.reshape(-1, 3)
        
        #adding a second receiver
        rx2 = ground_center + np.array([
            [35, 5, 5],
            [39, -5, 5],
            [10, -3, -5]
        ])
        rx2 = rx2[2, :]
        rx2 = rx2.reshape(-1, 3)
        
        
    elif geometry == 'dummy': #TODO: doesn't work because not preprocessed
        tx = np.array([5., 12., 5.]).reshape(1, 3)
        rx = np.array([65., 12., 5.]).reshape(1, 3)

        square_1 = geom.Square.by_2_corner_points(np.array([[13, 22, 0], [17, 24, 0]]))
        building_1 = geom.Building.by_polygon_and_height(square_1, 10,1)
        square_2 = geom.Square.by_2_corner_points(np.array([[33, 0, 0], [37, 2, 0]]))
        building_2 = geom.Building.by_polygon_and_height(square_2, 10,2)
        square_3 = geom.Square.by_2_corner_points(np.array([[53, 22, 0], [57, 24, 0]]))
        building_3 = geom.Building.by_polygon_and_height(square_3, 10,3)

        ground = geom.Square.by_2_corner_points(np.array([[0, 0, 0], [70, 24, 0]]))

        place = geom.OrientedPlace(geom.OrientedSurface(ground), [building_1, building_2, building_3])
        #dummy_filename=place.to_json(filename="../data/dummy.json")
        #geom.preprocess_geojson(dummy_filename)
        
        
    elif geometry == 'my_geometry':
        geometry_filename="../data/TutorialTest/sciences.geojson"
        
        geom.preprocess_geojson(geometry_filename)
        geometry_filename=geom.sample_geojson(geometry_filename,nBuildings=10)
        place = geom.generate_place_from_rooftops_file(geometry_filename)

        # 2. Create TX and RX

        domain = place.get_domain()
        ground_center = place.get_centroid()

        tx = ground_center + [-50, 5, 1]
        rx = ground_center + np.array([
            [35, 5, 5],
            [35, -5, 5],
            [10, -3, -5]
        ])
        rx = rx[2, :]
        tx = tx.reshape(-1, 3)
        rx = rx.reshape(-1, 3)
        
    
    # Adding receivers to place
    place.add_set_of_points(rx)
    #place.add_set_of_points(rx2)
    
    # This plot is here to check that you geometry is correct. (TODO doesn't work)
    #ax = place.plot3d(ret=True)
    #place.center_3d_plot(ax)
    #ax=plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX")
    #plt.legend()
    #plt.show()       
    
    #compute the rays
    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=3)
    fig = plt.figure()
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    ax = problem.plot3d(ax=ax)
    problem_path='../data/problem.json'
    problem.save(problem_path)
    
    
    #updated_problem=problem.load(problem_path)
    
    #compute E field
    #compute_field_from_solution(updated_problem, '../data/electromagnetism.json')
    
    my_field_computation(problem)
    
    #all_receivers=problem.receivers
    #the_emitter=problem.emitter
    #problem.reflections[0][1] "first  order reflections reaching the first receiver"
    #rtp.reflections[0][1][0] One of the first order reflection reaching the first receiver
    #print(rtp.diffractions[0][2]) order 2 reflections reaching the first receiver"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:56:31 2022

@author: max
"""

#self written imports
from ray_tracing import RayTracingProblem
import raytracing.geometry as geom
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data


#packages
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import matplotlib.pyplot as plt



N_CPU=16


    
def merge_solved_problems(full_problem,solved_problems,save_path):
    #merge solved problems into one    
    for problem in solved_problems:
        #get solved problem index
        all_receivers=problem.place.set_of_points
        solved_receiver=problem.solved_receivers[0]
        for i in range(len(all_receivers)):
            if all(all_receivers[i]==solved_receiver):
                index=i
        full_problem.reflections[index]=problem.reflections[index]
        full_problem.diffractions[index]=problem.diffractions[index] 
    full_problem.solved_receivers=full_problem.place.set_of_points
    full_problem.save(save_path)




if __name__ == "__main__":   
    plt.close('all')
    
    
    #init problem
    N_points_small=20
    N_points_levant=5
    geometry = 'small' #MAKE SURE TO RESTART KERNEL BEFORE EVERY RUN, OTHERWISE IT CRASHES!
    
   
    if geometry == 'small':
        geometry_filename='../data/small.geojson' 
        geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(geometry_filename)
        
        #add tx 
        ground_center = place.get_centroid()
        tx = ground_center + [-50, 5, 1]
        tx=tx.reshape(1,3)

        #add rx
        N_points=N_points_small
        rx=ground_center
        trans=np.array([10,5,0]).reshape(1,3)
        for i in range(N_points):
            place.add_set_of_points(rx+trans*i)
       
    
    elif geometry == 'levant':
        geometry_filename='../data/place_levant.geojson'
        geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(geometry_filename) 
        # 2. Create TX and RX
        tx = np.array([110,-40,35]).reshape(1, 3)
        rx = np.array([-30,-46,2]).reshape(1, 3)
        N_points=N_points_levant #how many rx to compute
        for i in range(0,N_points):
            rx +=np.array([20,0,0])
            place.add_set_of_points(rx)
    
    
    def compute_single_receiver(rx):     
        print(f"rx {rx}")
        problem = RayTracingProblem(tx, place)
        problem.solve(max_order=3,receivers_indexs=[rx])
        #problem_path=f"../data/problem{rx}.json"
        #problem.save(problem_path)
        return problem
    
    
    #solve each receiver on separate CPU
    pool = Pool(N_CPU)
    pool.restart()
    args=np.arange(0,len(place.set_of_points),1)
    solved_problems=pool.map(compute_single_receiver, range(N_points))
    pool.close() 
    pool.join()
  
    
    #merge the problems
    full_problem = RayTracingProblem(tx, place)
    merge_solved_problems(full_problem, solved_problems, "../data/merged_problem.json")
        
    
    
    #plot full problem
    fields=my_field_computation(full_problem)
    EM_fields_plots(fields)
    EM_fields_data(fields)
    full_problem.plot_rays()   
        
        
            
        
    

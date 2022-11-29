#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:56:31 2022
@author: Maxime Leurquin

USAGE:
MAKE SURE TO RESTART KERNEL BEFORE EVERY RUN, OTHERWISE IT CRASHES!
"""

#self written imports
from ray_tracing import RayTracingProblem
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data
import place_utils

#packages
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt


def multithread_solve_place(place,tx,save_name,N_CPU=16,ORDER=3):
    """
    Solves the place, using one CPU for each receiver.
    Saves the plots and dataframe in the results folder.
    place: place with all the receivers added
    """
    def compute_single_receiver(rx):
        """
        Solves the problem for the specified receiver index
        """
        print(f"solving rx {rx}")
        problem = RayTracingProblem(tx, place)
        problem.solve(max_order=ORDER,receivers_indexs=[rx])
        return problem
    def merge_solved_problems(full_problem,solved_problems,save_name):
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
        save_path=f"../results/{save_name}_ray_solved.json"
        full_problem.save(save_path)
        return save_path
    
    #solve each receiver on separate CPU
    pool = Pool(N_CPU)
    pool.restart()
    N_points=len(place.set_of_points)
    indices=range(N_points)
    solved_problems=pool.map(compute_single_receiver,indices) 
    pool.close() 
    pool.join()
    
    solved_em_path=f'../results/{save_name}_em_solved.csv'
    
    #merge the problems
    full_problem = RayTracingProblem(tx, place)
    solved_rays_path=merge_solved_problems(full_problem, solved_problems, save_name)
    full_problem.plot_all_rays()    
    #compute fields
    my_field_computation(full_problem,solved_em_path)
    
    #plot full problem
    EM_fields_plots(solved_em_path,order=ORDER,name=save_name)
    EM_fields_data(solved_em_path)
    
         
    return solved_em_path,solved_rays_path

if __name__ == "__main__":  
    #To test if it works
    plt.close('all')
    place,tx,geometry=place_utils.create_small_place(npoints=30)
    #place,tx,geometry=place_utils.create_levant_place(npoints=15)
    
    multithread_solve_place(place, tx,"multithread_place_test")
        
        
        
    
        
    

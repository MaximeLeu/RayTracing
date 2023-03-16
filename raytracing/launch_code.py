# -*- coding: utf-8 -*-
#pylint: disable=invalid-name,line-too-long
"""
Created on Thu Sep 29 18:33:28 2022

Code driver
@author: maxime
"""
#packages
import matplotlib.pyplot as plt

#self written imports
from ray_tracing import RayTracingProblem
from electromagnetism import my_field_computation,EM_fields_plots
from multithread_solve import multithread_solve_place
import file_utils
import place_utils
import raytracing.geometry as geom
import numpy as np

if __name__ == '__main__':
    plt.close('all') #close all previous plots
    file_utils.chdir_to_file_dir(__file__)

    #choose geometry:
    #place,tx,geometry=place_utils.create_two_rays_place()
    #place,tx,geometry=place_utils.create_dummy_place()
    #place,tx,geometry=place_utils.create_my_geometry() # random geometry
    #place,tx,geometry=place_utils.create_small_place()
    #place,tx,geometry=place_utils.create_levant_place()
    place,tx,geometry=place_utils.create_slanted_place(10)
    ORDER=2
    multithread=False #if true restart kernel before launching.
    
    #single threaded driver code
    if not multithread:
        #compute the rays
        problem = RayTracingProblem(tx, place)
        problem.solve(max_order=ORDER,receivers_indexs=None)
        problem_path=f'../results/{geometry}_ray_solved.json'
        problem.save(problem_path)
        problem.plot_all_rays()

        #compute the fields
        results_path=f'../results/{geometry}_em_solved.csv'
        df=my_field_computation(problem,results_path)

        #plots
        EM_fields_plots(results_path,order=ORDER,name=geometry)

    #multithreaded driver code
    if multithread:
        solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='{geometry}',order=ORDER)
        df=file_utils.load_df(solved_em_path)
        
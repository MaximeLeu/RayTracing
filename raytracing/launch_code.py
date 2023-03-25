# -*- coding: utf-8 -*-
#pylint: disable=invalid-name,line-too-long
"""
Created on Thu Sep 29 18:33:28 2022

Code driver
@author: Maxime Leurquin
"""
#packages
import matplotlib.pyplot as plt
import numpy as np

#self written imports
from ray_tracing import RayTracingProblem
from raytracing.electromagnetism import my_field_computation,EM_fields_plots
from raytracing.multithread_solve import multithread_solve_place
from raytracing.file_utils import chdir_to_file_dir, load_df

import raytracing.place_utils as place_utils
import raytracing.geometry as geom


if __name__ == '__main__':
    plt.close('all') #close all previous plots
    chdir_to_file_dir(__file__)

    #choose geometry:
    #place,tx,geometry=place_utils.create_two_rays_place()
    #place,tx,geometry=place_utils.create_dummy_place()
    #place,tx,geometry=place_utils.create_my_geometry() # random geometry
    #place,tx,geometry=place_utils.create_small_place()
    #place,tx,geometry=place_utils.create_levant_place()
    place,tx,geometry=place_utils.create_slanted_place(10)
    place_utils.plot_place(place, tx)
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
        df=load_df(solved_em_path)
        
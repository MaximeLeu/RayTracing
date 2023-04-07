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
from raytracing.ray_tracing import RayTracingProblem
from raytracing.electromagnetism import my_field_computation
import raytracing.electromagnetism_plots as electromagnetism_plots
from raytracing.multithread_solve import multithread_solve_place

import raytracing.electromagnetism_utils as electromagnetism_utils
import raytracing.place_utils as place_utils
import raytracing.file_utils as file_utils


if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close('all') #close all previous plots

    #choose geometry:
    #place,tx,geometry=place_utils.create_two_rays_place()
    #place,tx,geometry=place_utils.create_dummy_place()
    #place,tx,geometry=place_utils.create_my_geometry() # random geometry
    place,tx,geometry=place_utils.create_small_place(npoints=5)
    #place,tx,geometry=place_utils.create_flat_levant(npoints=3)
    #place,tx,geometry=place_utils.create_slanted_levant(10)
    place_utils.plot_place(place, tx)
    ORDER=4
    
    #if multithreading run the script outside spyder/Ipython, in a command prompt. 
    multithread=True 
    
    #single threaded driver code
    if not multithread:
        #compute the rays
        problem = RayTracingProblem(tx, place)
        problem.solve(max_order=ORDER,receivers_indexs=None)
        problem_path=f'../results/{geometry}_ray_solved.json'
        problem.to_json(problem_path)
        problem.plot_all_rays()

        #compute the fields
        solved_em_path=f'../results/{geometry}_em_solved.csv'
        my_field_computation(problem,solved_em_path)

        #plots
        electromagnetism_plots.EM_fields_plots(solved_em_path,order=ORDER,name=geometry)
        

    #multithreaded driver code
    if multithread:
        solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,order=ORDER,geometry=geometry)
        
        df=electromagnetism_utils.load_df(solved_em_path)
        npoints=len(df['rx_id'].unique())
        save_name=f'{geometry}_{npoints}p'
        electromagnetism_plots.EM_fields_plots(solved_em_path,order=ORDER,name=save_name)
        
        
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
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data
import file_utils
import place_utils





#single threaded driver code
if __name__ == '__main__':
    plt.close('all') #close all previous plots
    file_utils.chdir_to_file_dir(__file__)

    #choose geometry:
    #place,tx,geometry=place_utils.create_two_rays_place()
    #place,tx,geometry=place_utils.create_dummy_place()
    #place,tx,geometry=place_utils.create_my_geometry() # random geometry
    place,tx,geometry=place_utils.create_small_place()
    #place,tx,geometry=place_utils.create_levant_place()

    solve=True
    ORDER=3

    if solve==True:
        #compute the rays
        problem = RayTracingProblem(tx, place)
        problem.solve(max_order=ORDER,receivers_indexs=None)
        problem_path=f'../results/{geometry}_launch.json'
        problem.save(problem_path)

        #compute the fields
        results_path=f'../results/{geometry}_launch.csv'
        df=my_field_computation(problem,results_path)

        #plots
        EM_fields_plots(results_path,order=ORDER,name=geometry)
        EM_fields_data(results_path)

        problem.plot_all_rays()














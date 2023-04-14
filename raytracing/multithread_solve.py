#pylint: disable=invalid-name,line-too-long
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:56:31 2022
@author: Maxime Leurquin

USAGE:
MAKE SURE TO RESTART KERNEL BEFORE EVERY RUN IF USING IPYTHON.
No problem when running in a command prompt.
"""
import time
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt


#self written imports
from raytracing.ray_tracing import RayTracingProblem
from raytracing.electromagnetism import my_field_computation
from raytracing.electromagnetism_plots import EM_fields_plots

import raytracing.place_utils as place_utils
import raytracing.file_utils as file_utils



def print_elapsed_time(start_time,end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {minutes} minutes {seconds} seconds\n")
    return


def compute_single_receiver(arg):
    """
    Solves the problem for the specified receiver index
    arg should be [rx_id,solve_order,tx_coords,place]
    """
    rx,order,tx,place=arg
    print(f"solving rx {rx}")
    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=order,receivers_indexs=[rx])
    return problem

def merge_solved_problems(full_problem,solved_problems,save_name):
    #merge solved problems into one
    for problem in solved_problems:
        #get solved problem index
        all_receivers=problem.place.set_of_points
        solved_receiver=problem.solved_receivers[0]
        for ind, receiver in enumerate(all_receivers):
            if all(receiver==solved_receiver):
                index=ind
        full_problem.reflections[index]=problem.reflections[index]
        full_problem.diffractions[index]=problem.diffractions[index]
    full_problem.solved_receivers=full_problem.place.set_of_points  
    
    save_path=f"../results/{save_name}_ray_solved.json"
    full_problem.to_json(save_path)
    return save_path

def multithread_solve_place(place,tx,geometry,N_CPU=16,order=3,save_name=None):
    """
    Solves the place, using one CPU for each receiver.
    Saves the plots and dataframe in the results folder.
    place: place with all the receivers added
    """
    print("Starting solving of the ray tracing problems")
    npoints=len(place.set_of_points)
    if save_name is None:
        save_name=f'{geometry}_{npoints}p'
    
    start_time_rt = time.time()
    #distribute the problems
    with Pool(N_CPU) as pool:
        pool.clear()
        args=[]
        for i in range(npoints):
            args.append([i,order,tx,place])
        solved_problems=pool.map(compute_single_receiver,args)
        pool.close()
        pool.join()

    #merge the solved_problems into one full_problem
    full_problem = RayTracingProblem(tx, place)
    solved_rays_path=merge_solved_problems(full_problem, solved_problems, save_name)
    end_time_rt = time.time()
    print("Finished solving the ray tracing problems")
    print_elapsed_time(start_time_rt,end_time_rt)
    full_problem.plot_all_rays()

    print("Starting EM solving")
    start_time_em = time.time()
    #compute fields
    solved_em_path=f'../results/{save_name}_em_solved.csv'
    my_field_computation(full_problem,solved_em_path)
    end_time_em = time.time()
    print("Finished EM solving and saved results")
    print_elapsed_time(start_time_em,end_time_em)
    print("TOTAL TIME: ")
    print_elapsed_time(start_time_rt,end_time_em)
    

    # print("Plotting EM plots:")
    # start_time_plt = time.time()
    # #plot full problem
    # EM_fields_plots(solved_em_path,order=order,name=save_name)
    # end_time_plt = time.time()
    # print("Finished EM plotting")
    # print_elapsed_time(start_time_plt,end_time_plt)
    
    
    
    return solved_em_path,solved_rays_path




# #works consecutively but 4 times slower...
#from concurrent.futures import ThreadPoolExecutor
# def multithread_solve_place(place, tx, save_name, N_CPU=16, order=3):
#     """
#     Solves the place, using one CPU for each receiver.
#     Saves the plots and dataframe in the results folder.
#     place: place with all the receivers added
#     """
#     # distribute the problems
#     args = []
#     for i in range(len(place.set_of_points)):
#         args.append([i, order, tx, place])
#     # Create a ThreadPoolExecutor to handle the parallel tasks
#     with ThreadPoolExecutor(max_workers=N_CPU) as executor:
#         # Use the executor.map function to apply 'compute_single_receiver' on each set of arguments
#         solved_problems = list(executor.map(compute_single_receiver, args))
#     # merge the solved_problems into one full_problem
#     full_problem = RayTracingProblem(tx, place)
#     solved_rays_path = merge_solved_problems(full_problem, solved_problems, save_name)
#     full_problem.plot_all_rays()
#     #compute fields
#     solved_em_path=f'../results/{save_name}_em_solved.csv'
#     my_field_computation(full_problem,solved_em_path)

#     #plot full problem
#     EM_fields_plots(solved_em_path,order=order,name=save_name)
#     return solved_em_path,solved_rays_path
        

    

if __name__ == "__main__":
    file_utils.chdir_to_file_dir(__file__)
    #To test if it works
    plt.close('all')
    place,tx,geometry=place_utils.create_small_place(npoints=15)
    place,tx,geometry=place_utils.create_slanted_dummy(npoints=15)
    #place,tx,geometry=place_utils.create_levant_place(npoints=15)
    multithread_solve_place(place, tx,"multithread_place_test")
    
    
    
    
    

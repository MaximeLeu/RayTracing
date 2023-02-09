#pylint: disable=invalid-name,line-too-long
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:56:31 2022
@author: Maxime Leurquin

USAGE:
MAKE SURE TO RESTART KERNEL BEFORE EVERY RUN, OTHERWISE IT CRASHES!
"""
#packages
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt


#self written imports
from ray_tracing import RayTracingProblem
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data
import place_utils



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
    full_problem.save(save_path)
    return save_path

def multithread_solve_place(place,tx,save_name,N_CPU=16,order=3):
    """
    Solves the place, using one CPU for each receiver.
    Saves the plots and dataframe in the results folder.
    place: place with all the receivers added
    """
    #distribute the problems
    pool = Pool(N_CPU)
    pool.restart()
    args=[]
    for i in range(len(place.set_of_points)):
        args.append([i,order,tx,place])
    solved_problems=pool.map(compute_single_receiver,args)
    pool.close()
    pool.join()

    #merge the solved_problems into one full_problem
    full_problem = RayTracingProblem(tx, place)
    solved_rays_path=merge_solved_problems(full_problem, solved_problems, save_name)
    full_problem.plot_all_rays()

    #compute fields
    solved_em_path=f'../results/{save_name}_em_solved.csv'
    my_field_computation(full_problem,solved_em_path)

    #plot full problem
    #EM_fields_plots(solved_em_path,order=order,name=save_name)
    EM_fields_data(solved_em_path)
    return solved_em_path,solved_rays_path


if __name__ == "__main__":
    #To test if it works
    plt.close('all')
    place,tx,geometry=place_utils.create_small_place(npoints=15)
    #place,tx,geometry=place_utils.create_levant_place(npoints=15)

    multithread_solve_place(place, tx,"multithread_place_test")

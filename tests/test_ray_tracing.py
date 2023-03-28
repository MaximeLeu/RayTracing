#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:38:07 2023

@author: Maxime Leurquin
"""
import numpy as np
from raytracing.ray_tracing import RayTracingProblem
from raytracing.place_utils import create_small_place,create_flat_levant

import raytracing.file_utils as file_utils
file_utils.chdir_to_file_dir(__file__)

def test_save_load_raytracingProblem():
    place,tx,geometry=create_small_place(npoints=3)
    problem=RayTracingProblem(tx, place)
    idx=np.array(range(len(place.set_of_points)))
    problem.solve(max_order=2,receivers_indexs=idx)
    
    problem.to_json(filename="saved_raytracing_problem.json")
    
    loaded_problem=RayTracingProblem.from_json(filename='saved_raytracing_problem.json')

    assert(problem==loaded_problem),"problems are not equal"
    print("")
    print("")
    print("TEST SUCCESS")
    return problem,loaded_problem
    
    

if __name__ == '__main__':
    problem,loaded_problem=test_save_load_raytracingProblem()
    

    
    
    
    
    
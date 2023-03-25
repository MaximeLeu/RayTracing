#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:16:53 2023

@author: Maxime Leurquin
"""

import numpy as np
import raytracing.geometry as geom
import timeit

def test_project_points():
    np.random.seed(42)
    
    for _ in range(10):
        points = np.random.rand(5, 3)
        matrix = np.random.rand(3, 3)
        around_point = np.random.rand(1, 3)
        
        # Test without around_point
        prev_result1 = geom.previous_project_points(points, matrix)
        new_result1 = geom.project_points(points, matrix)
        assert np.allclose(prev_result1, new_result1), "Results do not match without around_point"
        
        # Test with around_point
        prev_result2 = geom.previous_project_points(points, matrix, around_point)
        new_result2 = geom.project_points(points, matrix, around_point)
        assert np.allclose(prev_result2, new_result2), "Results do not match with around_point"


def time_functions():
    setup = '''
import numpy as np
import geometry as geom

np.random.seed(42)
points = np.random.rand(5, 3)
matrix = np.random.rand(3, 3)
around_point = np.random.rand(1, 3)
    '''

    previous_code = '''
geom.previous_project_points(points, matrix)
geom.previous_project_points(points, matrix, around_point)
    '''

    new_code = '''
geom.project_points(points, matrix)
geom.project_points(points, matrix, around_point)
    '''

    num_iterations = 100000

    prev_time = timeit.timeit(previous_code, setup=setup, number=num_iterations)
    new_time = timeit.timeit(new_code, setup=setup, number=num_iterations)

    print(f"Previous implementation time: {prev_time:.5f} seconds")
    print(f"New implementation time: {new_time:.5f} seconds")




if __name__ == '__main__':
    print("hello")
    test_project_points()
    print('hhh')
    time_functions()
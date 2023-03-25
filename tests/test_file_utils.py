#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:58:01 2023
@author: Maxime Leurquin
Testing of functions defined in the file_utils file
"""
import numpy as np
import pandas as pd

from raytracing.file_utils import string_arr_to_ndarray, string_complex_arr_to_ndarray,\
         load_df,save_df





def test_string_arr_to_ndarray():
    # Test the string_arr_to_ndarray function
    input_strings = ['[1, 2, 3]', '[4.5, 6.7, 8.9]', '[-1, 0, 1]']
    expected_results = [np.array([1, 2, 3]), np.array([4.5, 6.7, 8.9]), np.array([-1, 0, 1])]
    for i, string in enumerate(input_strings):
        result = string_arr_to_ndarray(string)
        expected = expected_results[i]
        assert np.array_equal(result, expected), f"Failed for string {string}. Expected {expected}, but got {result}."
    print("Test success for string_arr_to_ndarray")
    return

def test_string_complex_arr_to_ndarray():
    # Test the string_complex_arr_to_ndarray function
    input_strings = ['[(1+1j), (2+2j), (3+3j)]', '[(1-1j), (2-2j), (3-3j)]', '[(0+0j), (0+1j), (1+0j)]']
    expected_results = [np.array([1+1j, 2+2j, 3+3j]), np.array([1-1j, 2-2j, 3-3j]), np.array([0+0j, 0+1j, 1+0j])]
    for i, string in enumerate(input_strings):
        result = string_complex_arr_to_ndarray(string)
        expected = expected_results[i]
        assert np.array_equal(result, expected), f"Failed for string {string}. Expected {expected}, but got {result}."
    print("Test success for string_complex_arr_to_ndarray")
    return


def create_test_df():
    # Create a dataframe similar to the one created in the electromagnetism.my_field_computation function
    df=pd.DataFrame(columns=['rx_id','receiver','path_type','time_to_receiver','field_strength','path_power'])
    df["rx_id"]=[0,0,1]
    df['receiver']=[np.array([10.1,10.1,1.2]),
                np.array([10.1,10.1,1.2]),
                np.array([20,20,1.2])]
    df['path_type']=['LOS','R','LOS']
    df['time_to_receiver']=[1.9*1e-9,1.78*1e-7,1.17*1e-9]
    df['field_strength']=[np.array([-0.081+0.016j,0.+0.j,0.315-0.06j]),
                       np.array([-0.0055-0.0103j,-0.+0.j,-0.016-0.03j]),
                       np.array([1.290*1e-02+1j*1.53,-4*1e-5-1j*5.7*1e-5,-5.0*1e-2-1j*5.9e-2])]
    df['path_power']=[0.0005,7.41309*1e-7,7.17*1e-6]
    return df


def test_save_and_load_df():
    # Test the save_df and load_df functions
    df=create_test_df()
    save_name = 'test_save_and_load_df.csv'
    save_df(df, save_name)
    loaded_df = load_df(f'../results/{save_name}')
    #Compare the original DataFrame to the loaded one
    pd.testing.assert_frame_equal(df, loaded_df,check_exact=True)
    print("Test success for save_df and load_df functions")
    return



if __name__ == '__main__':
    test_string_complex_arr_to_ndarray()
    test_string_arr_to_ndarray()
    test_save_and_load_df()
    print("Success of all tests")
    
    
    
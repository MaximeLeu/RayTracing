#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:33:43 2023

@author: max
"""
import numpy as np
import scipy as sc
import pandas as pd
from scipy.constants import pi


from raytracing.materials_properties import P_IN,\
                                RADIATION_EFFICIENCY,\
                                RADIATION_POWER,\
                                ALPHA,\
                                TX_GAIN,\
                                RX_GAIN,\
                                FREQUENCY,\
                                LAMBDA,\
                                K,\
                                Z_0\


def to_spherical(point):
    """
    transform the given point into spherical coordinates
    """
    x,y,z=point
    if np.all(point==0):
        assert False,"spherical coordinates system not defined for point at the origin."
    if x==0 and y==0:
        assert False,"spherical coordinates system not defined for points on z axis"
    hxy=np.sqrt(x**2+y**2) #proj of r on plane xy
    r = np.sqrt(hxy**2+z**2)
    el = np.arctan2(hxy, z) #from z to ray
    az = np.arctan2(y, x) #from x to proj ray on plane xy
    return r,el,az


def to_cartesian(r,theta,phi):
    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def cot(x):
    return 1/np.tan(x)

def vv_normalize(vv):
    norm=np.linalg.norm(vv)
    if norm == 0:#avoid dividing by 0
        return np.array([0,0,0])
    return vv/norm

def to_db(field_power):
    """
    converts given field power in watts to dB (normalised to INPUT_POWER)
    """
    #avoid computing log(0)
    P=np.atleast_1d(field_power)
    ind=np.where(P!=0)
    db=np.ones(len(P))*np.NINF
    for i in ind:
        db[i]=10*np.log10(P[i]/P_IN)
    return db[0] if np.isscalar(field_power) else db

def string_arr_to_ndarray(string_arr):
    """
    Converts a string representation of an array to a numpy ndarray.
    :param string_arr: The string representation of the array, in the form "[1, 2, 3]".
    :type string_arr: str
    :return: The converted ndarray
    :rtype: numpy.ndarray
    """
    if not isinstance(string_arr, np.ndarray):
        string_arr=np.fromstring(string_arr[1:-1],sep=',')
        #assert len(string_arr)==3, f"strange array shape: {string_arr}"     
    return string_arr

def string_complex_arr_to_ndarray(string):
    """
    Converts a string representation of an array of complex numbers to a numpy ndarray.
    :param string_arr: The string representation of the array, in the form "[(1+1j) ]".
    :type string_arr: str
    :return: The converted ndarray
    :rtype: numpy.ndarray
    """
    #Remove the brackets and split the string into components
    split=string[1:-1].split(',')
    assert len(split)==3, f"strange array shape: split: {split}"
    # Convert each component to a complex number and store in a numpy ndarray
    return np.array([complex(c) for c in split])


def save_df(df,save_name):
    """
    Save a pandas DataFrame to a CSV file with the specified filename in the '../results/save_name.csv' directory.
    :param df: pandas DataFrame to save
    :type df: pandas.DataFrame
    :param save_name: The file save_name
    :type save_name: str
    :return: None
    :rtype: None
    """
    if not save_name.endswith('.csv'):
        save_name += '.csv'
    #convert to strings for easier loading 
    df=df.copy(deep=True)
    for i in range(len(df['field_strength'])):
        string=np.array2string(df.at[i,'field_strength'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'field_strength']=string 
        
        string=np.array2string(df.at[i,'receiver'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'receiver']=string
        
        string=np.array2string(df.at[i,'tx_angles'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'tx_angles']=string
        
        string=np.array2string(df.at[i,'rx_angles'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'rx_angles']=string
        
    df.to_csv(f"../results/{save_name}",index=False)

def load_df(df_path):
    """
    Load a pandas DataFrame from a CSV file.
    :param df_path: The path to the CSV file to load.
    :type df_path: str
    :return: The loaded DataFrame.
    :rtype: pandas.DataFrame
    """
    df=pd.read_csv(df_path)
    #converts string to ndarrays
    for i in range(len(df["receiver"].values)):
        df.at[i,"receiver"]=string_arr_to_ndarray(df.at[i,"receiver"])
        df.at[i,"field_strength"]=string_complex_arr_to_ndarray(df.at[i,"field_strength"])
    
        df.at[i,'tx_angles']=string_arr_to_ndarray(df.at[i,'tx_angles'])
        df.at[i,'rx_angles']=string_arr_to_ndarray(df.at[i,'rx_angles'])
        
    return df
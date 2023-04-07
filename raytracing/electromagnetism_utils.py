#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:33:43 2023

@author: max
"""
import numpy as np
import pandas as pd


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


def generate_path_types_of_order(order):
    """
    Generates the possible path types for the given order
    """
    if order == 0:
        return ["LOS"]
    ref="R"*order
    diff="R"*(order-1)+"D"
    return [ref,diff]

def generate_path_types_up_to_order(order):
    """
    Generates the possible path types up to the given order
    """
    array=[]
    for i in range(0,order+1):
        array.extend(generate_path_types_of_order(i))
    return array

def find_df_order(df):
    """
    Finds what is the highest order path in the dataframe
    """
    for i in range(1,20):
       possible_paths=generate_path_types_of_order(i)
       if not df['path_type'].isin(possible_paths).any():
           return i-1
    assert False
        
def get_data_up_to_order(df,order):
    """
    return a dataframe truncated to only contain data up until order
    ex: get_data_up_to_order(2) will return all LOS,R,RR,RD
    get_data_up_to_order(0) will return LOS
    """  
    all_types=generate_path_types_up_to_order(order)
    selected_types=all_types[:(3+(order-1)*2)]
    df_filtered = df[df['path_type'].isin(selected_types)]
    return df_filtered


def get_power_db_each_receiver(df):
    """
    Compute the total power at each receiver by summing
    all the individual power contributions, then converting to dB
    """
    receivers=df['rx_id'].unique()
    power_db=np.zeros(len(receivers))
    for i,receiver in enumerate(receivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        power_db[i]=to_db(np.sum(rx_df["path_power"]))
    return power_db


def split_df(df,df_list=[]):
    """
    Split the df containing information about the fields transmitted to each receiver
    to a list of dataframes containing data about 50 receivers each.
    """
    nreceivers=len(df['rx_id'].unique())
    if nreceivers>50:
        first_50_receivers = df['rx_id'].unique()[:50]
        first_50_df = pd.concat([df.loc[df['rx_id'] == receiver] for receiver in first_50_receivers])
        rest_df = df.loc[~df['rx_id'].isin(first_50_receivers)]
        df_list.append(first_50_df)
        split_df(rest_df,df_list)
    else:
        df_list.append(df)
    return df_list


def to_spherical(point):
    """
    transform the given point into spherical coordinates
    """
    x,y,z=point
    if np.all(point==0):
        assert False,"spherical coordinates system not defined for point at the origin."
    if x==0 and y==0:
        print("WARNING: spherical coordinates system not defined for points on z axis, azimutal angle can be anything")
        return z,0,0
       
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
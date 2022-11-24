# Utils
import os
import json
import numpy as np
import pandas as pd

def json_save(filename, data, *args, indent=4, **kwargs):
    """
    Writes a dictionary into a json format file.

    :param filename: the filepath
    :type filename: str
    :param data: the dictionary
    :type data: dict
    :param args: positional arguments to be passed to :func:`json.dump`
    :type args: any
    :param indent: indentation size
    :type indent: int
    :param kwargs: keyword arguments to be passed to :func:`json.dump`
    """
    with open(filename, 'w') as f:
        json.dump(data, f, *args, indent=indent, **kwargs)


def json_load(filename, *args, **kwargs):
    """
    Read a dictionary from a json format file.

    :param filename: the filepath
    :type filename: str
    :param args: positional arguments to be passed to :func:`json.load`
    :type args: any
    :param kwargs: keyword arguments to be passed to :func:`json.load`
    :return: the dictionary
    :rtype: dict
    """
    with open(filename, 'r') as f:
        return json.load(f, *args, **kwargs)


def chdir_to_file_dir(filename):
    """
    Changes the working directory to be the same as the one containing filename.

    :param filename: the filename
    :type filename: str

    :Example:

    >>> os.getcwd()
    '/pathto/RadarCoverage'
    >>> chdir_to_file_dir(__file__)
    >>> os.getcwd()
    '/pathto/RadarCoverage/raytracing'
    """
    os.chdir(os.path.dirname(os.path.realpath(filename)))

def save_df(df,save_name):
    #convert to strings for easier loading 
    df=df.copy(deep=True)
    for i in range(len(df['field_strength'])):
        string=np.array2string(df.at[i,'field_strength'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'field_strength']=string
        
        string=np.array2string(df.at[i,'receiver'],separator=",",max_line_width=999)
        string=string.replace(" ","")
        df.at[i,'receiver']=string
    df.to_csv(f"../results/{save_name}",index=False)

def load_df(df_path):
    def string_arr_to_ndarray(string_arr):
        if not isinstance(string_arr, np.ndarray):
            #converts "[1,2,3]" to an nparray
            string_arr=np.fromstring(string_arr[1:-1],sep=',')
            assert len(string_arr)==3     
        return string_arr

    def string_complex_arr_to_ndarray(string):
        #converts "[(1+1j) ]"
        split=string[1:-1].split(',')
        assert len(split)==3, f"split: {split}"
        vector=np.zeros(3,dtype=complex)
        for i in range(3):
            vector[i]=complex(split[i])
        return vector   
    
    df=pd.read_csv(df_path)
    #converts string to ndarrays
    for i in range(len(df["receiver"].values)):
        df.at[i,"receiver"]=string_arr_to_ndarray(df.at[i,"receiver"])
        df.at[i,"field_strength"]=string_complex_arr_to_ndarray(df.at[i,"field_strength"])
    return df



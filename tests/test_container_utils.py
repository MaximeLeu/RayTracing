#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:42:43 2023

@author: Maxime Leurquin
"""
import numpy as np
import pickle

from raytracing.container_utils import ManyToOneDict
import raytracing.file_utils as file_utils


def create_dict():
    
    d1=ManyToOneDict()
    d1[(1,2)]=np.array([[1,1,1],[2,2,2]])
    d1[(1,3)]=np.array([[1,1,1],[3,3,3]])
    d1[(2,3)]=np.array([[2.,2,2],[3,3,3]])
    d1[(8,9)]=np.array([[8,8,8],[9,9,9]])
    d1[(8,10)]=np.array([[8,8,8],[10,10,10]])
    d1[(8,11)]=np.array([[8,8,8],[11,11,11]])
    
    #slightly different than d1
    d2=ManyToOneDict()
    d2[(1,2)]=np.array([[1,1,1],[2,2,2]])
    d2[(1,3)]=np.array([[1,1,1],[3,3,3]])
    d2[(2,3)]=np.array([[2,2,2],[3,3,3]])
    d2[(8,9)]=np.array([[8,8,8],[1999,1999,1999]])
    d2[(8,10)]=np.array([[8,8,8],[10,10,10]])
    d2[(8,11)]=np.array([[8,8,8],[11,11,11]])
    
    #same as d1
    d3=ManyToOneDict()
    d3[(1,2)]=np.array([[1,1,1],[2,2,2]])
    d3[(1,3)]=np.array([[1,1,1],[3,3,3]])
    d3[(2,3)]=np.array([[2,2,2],[3,3,3]])
    d3[(8,9)]=np.array([[8,8,8],[9,9,9]])
    d3[(8,10)]=np.array([[8,8,8],[10,10,10]])
    d3[(8,11)]=np.array([[8,8,8],[11,11,11]])
    return d1,d2,d3


def test_equals():
    d1,d2,d3=create_dict()
    assert d1==d3
    assert (d1==d2)==False
    print("test_equals success")
    return

def test_save_load():
    d1,_,_=create_dict()
    filename = "saved_many_to_one_dict.json"
    d1.to_json(filename=filename)
    d_new = ManyToOneDict.from_json(filename=filename)
    assert d1 == d_new
    print("test save_load success")

    
def test_pickling():
    d1,d2,d3=create_dict()
    
    pickled_data = pickle.dumps(d1)
    unpickled_dict = pickle.loads(pickled_data)
    assert d1==unpickled_dict
    print("pickling success")
    
if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    test_equals()
    test_save_load()
    test_pickling()
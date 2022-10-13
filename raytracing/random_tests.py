#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:54:36 2022

@author: maxime
"""
from ray_tracing import RayTracingProblem
import raytracing.geometry as geom
import numpy as np
from raytracing import plot_utils
import matplotlib.pyplot as plt
import json
from raytracing import file_utils
import geopandas as gpd
from shapely.geometry import Polygon as shPolygon
from shapely.geometry import Point as shPoint
from geometry import Building, OrientedSurface,OrientedPlace

filename="../data/TutorialTest/sciences.geojson"

arr=tuple([1,2])
print(arr)
lis=1,2
print(lis)
points=np.zeros((0, 3))
print(points.shape[1] == 3)
print(arr)
arr2=np.array([[10.0, -3.0, 15.0], [10.0, -3.0, 15.0]])
arr3=arr2[0]
print("dot")
a=np.array([1,1,1])
b=np.array([2,3,4])
print(b[0:-1])

c=[0,1,2,3,4,5]
print(c[-3:])

a=[]
a.append(1+2j)
a.append(14)
print(a)
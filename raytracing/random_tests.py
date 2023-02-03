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
from geometry import Building, OrientedSurface,OrientedPlace,OrientedPolygon,Square
import shapely
import random
import itertools
plt.close('all')





if __name__ == "__main__":
  a=np.array([np.array([1,2,3]),np.array([4,5,6])])
  
  b=9.117715180773833e-15
  assert(-1e-5<b<1e-5)
  
  a=["a","b","c","d","e"]
  for i in enumerate(a):
      print(i)

      
     
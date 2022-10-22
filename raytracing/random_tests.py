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


geometry_filename='../data/small.geojson'
geom.preprocess_geojson(geometry_filename)
original_place = geom.generate_place_from_rooftops_file(geometry_filename)


for i in range (0,10):
    original_place.add_tree(3)

#-------------------------------------------------------------
fig = plt.figure("original place and computed zone")
fig.set_dpi(300)
ax = fig.add_subplot(projection='3d')
original_place.center_3d_plot(ax)
ax = original_place.plot3d(ax=ax)    



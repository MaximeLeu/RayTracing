#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:19:25 2022

@author: maxime
"""
import raytracing.geometry as geom
from raytracing import file_utils
import matplotlib.pyplot as plt
import geopandas as gpd  # noqa
import numpy as np

if __name__ == '__main__':
    filename = '../data/TutorialTest/placeSciences.geojson'  
    gdf = gpd.read_file(filename)
    if not 'height' in gdf.columns:
        minHeight=25
        maxHeight=40
        print("Missing height data, adding random heights between {} m and {} m".format(minHeight,maxHeight))
        height=np.round(np.random.uniform(low=minHeight, high=maxHeight, size=len(gdf)),1)
        gdf['height']=height
        with open('dataframe.geojson' , 'w') as file:
            gdf.to_file(filename, driver="GeoJSON") 
    
    
    
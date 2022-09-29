"""
This file shows you how to plot the geometry in 3D.
"""

import raytracing.geometry as geom
from raytracing import file_utils
import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np
from pathlib import Path





if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)
    filename = '../data/TutorialTest/sciences.geojson' 
    #filename = '../data/ny.geojson'
    
    
    geom.preprocess_geojson(filename)
    filename=geom.sample_geojson(filename,nBuildings=10)
    
    
    place = geom.generate_place_from_rooftops_file(filename)

    ax = place.plot3d(ret=True,poly_kwargs=dict(orientation=True, normal=True))

    place.center_3d_plot(ax)
    place.get_sharp_edges(min_angle=10.0)
    place.show_sharp_edges_animation()
    place.show_visibility_matrix_animation(strict=False) #only shows once the previous animation is closed.
    
    plt.show()

#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:23:12 2023

@author: Maxime Leurquin
"""
#packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import shapely

#self written imports
from raytracing.multithread_solve import multithread_solve_place
import raytracing.ray_tracing as ray_tracing
import raytracing.geometry as geom
import raytracing.electromagnetism_plots as electromagnetism_plots


import raytracing.place_utils as place_utils
from place_utils import Place_saint_jean
import raytracing.plot_utils as plot_utils
import raytracing.electromagnetism_utils as electromagnetism_utils
import raytracing.file_utils as file_utils
plt.rcParams.update({'font.size': 20})


def radiomap(df,save_name,interpolation_method='linear'):
    #interpol method can be linear, cubic, or nearest
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Radiomap',fontsize=20)
    cmap = plt.get_cmap('jet')
    
    #get data
    receivers=electromagnetism_utils.get_receiver_coordinates(df,two_dimensionnal=True)
    powers=electromagnetism_utils.get_power_db_each_receiver(df)
    street=Place_saint_jean.select_saint_jean_streets_hardcoded()
    
    #plot buildings
    geometry_filename='../data/place_saint_jean/place_saint_jean.geojson'
    preprocessed_name=geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(preprocessed_name)
    for polyhedron in place.polyhedra:
        top=polyhedron.get_top_face().get_shapely()
        ax=plot_utils.plot_shapely(top,ax,color="Black")
    
    # Create a grid inside the street polygon
    grid_resolution = 300
    x, y = np.meshgrid(
        np.linspace(street.bounds[0], street.bounds[2], grid_resolution),
        np.linspace(street.bounds[1], street.bounds[3], grid_resolution)
    )

    # Interpolate power values on the grid
    z = griddata(receivers, powers, (x, y), method=interpolation_method, fill_value=np.min(powers))

    # Mask grid points outside the street polygon
    mask = np.array([street.contains(shapely.geometry.Point(x_, y_)) for x_, y_ in zip(x.ravel(), y.ravel())])
    z_masked = np.ma.masked_where(np.logical_not(mask.reshape(x.shape)), z)

    # Plot the colormap
    ax.pcolormesh(x, y, z_masked, cmap=cmap, shading='auto')
    fig.colorbar(ax.pcolormesh(x, y, z_masked, cmap=cmap, shading='auto'), ax=ax, label='Power (dB)')

    # Plot the receiver points and their power values
    for receiver, power in zip(receivers, powers):
        ax.plot(receiver[0], receiver[1], 'ko', markersize=4)
        ax.text(receiver[0], receiver[1], f"{power:.1f}", fontsize=8)
        
    plt.savefig(f"../results/plots/radiomap_{save_name}.eps", format='eps', dpi=300,bbox_inches='tight')
    plt.show()
    return





def run_saint_jean_simu(points_filename,order=2,flat=False):
    place,tx,geometry=Place_saint_jean.create_place_saint_jean(points_filename) if flat \
        else Place_saint_jean.create_slanted_place_saint_jean(points_filename)
    place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,geometry=geometry,order=order)
    return tx,solved_em_path,solved_rays_path



if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close("all")
    #points_filename="../data/place_saint_jean/place_saint_jean_160_points_d_4.json"
    # points_filename="../data/place_saint_jean/place_saint_jean_16_points_d_4.json"
   
    #electromagnetism_plots.plot_order_importance(solved_em_path)
    #electromagnetism_plots.EM_fields_plots(solved_em_path,name="place_saint_jean")
    
    # solved_rays_path="../results/place_saint_jean_16p_ray_solved.json"
    # problem=ray_tracing.RayTracingProblem.from_json(solved_rays_path)
    # problem.plot_specific_receiver(10)
    
    # points_filename="../data/place_saint_jean/place_saint_jean_16_points_d_4.json"
    # Place_saint_jean.plot_points(points_filename)
    
    # df=electromagnetism_utils.load_df(solved_em_path)
    # receivers=[1]
    # electromagnetism_plots.plot_rx_rays_distribution(df,receivers)
    # electromagnetism_plots.plot_rays_on_sphere_helper(df,receivers)
    
    
    def run_simu_suite(points_filename="../data/place_saint_jean/place_saint_jean_16_points_d_4.json"):
        receivers=[1]
        
        tx,solved_em_path,solved_rays_path=run_saint_jean_simu(points_filename,order=2,flat=False)
        df=electromagnetism_utils.load_df(solved_em_path)
        save_name='slanted_stjean' 
        radiomap(df,save_name,interpolation_method='linear')
        electromagnetism_plots.plot_rx_rays_distribution(df,receivers,save_name)
        electromagnetism_plots.plot_rays_on_sphere_helper(df,receivers,save_name)
         
        # save_name='flat_stjean' 
        # tx,solved_em_path,solved_rays_path=run_saint_jean_simu(points_filename,order=2,flat=True)
        # df=electromagnetism_utils.load_df(solved_em_path)
        # radiomap(df,save_name,interpolation_method='linear')
        # electromagnetism_plots.plot_rx_rays_distribution(df,receivers,save_name)
        # electromagnetism_plots.plot_rays_on_sphere_helper(df,receivers,save_name)
        
    run_simu_suite()
        
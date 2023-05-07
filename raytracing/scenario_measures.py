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
import time
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

def compare_radiomaps(dfs,txs,save_name=None):
    interpolation_method="linear"
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    txs = [txs] if not isinstance(txs, list) else txs
    fig, axes = plt.subplots(nrows=1,ncols=len(dfs),figsize=(15*len(dfs), 15))
    axes=[axes] if not isinstance(axes,np.ndarray) else axes
    
    place=Place_saint_jean.gen_place_saint_jean()
    # Create a grid inside the street polygon
    street=Place_saint_jean.select_saint_jean_streets_hardcoded()
    grid_resolution = 300
    x, y = np.meshgrid(
        np.linspace(street.bounds[0], street.bounds[2], grid_resolution),
        np.linspace(street.bounds[1], street.bounds[3], grid_resolution)
    )
    # Mask grid points outside the street polygon
    mask = np.array([street.contains(shapely.geometry.Point(x_, y_)) for x_, y_ in zip(x.ravel(), y.ravel())])
    
    cmap = plt.get_cmap('jet')
    for i, df in enumerate(dfs):
        axes[i].set_title('Radiomap',fontsize=20)
        #get data
        receivers=electromagnetism_utils.get_receiver_coordinates(dfs[i],two_dimensionnal=True)
        powers=electromagnetism_utils.get_power_db_each_receiver(dfs[i])
        #plot buildings
        for polyhedron in place.polyhedra:
            top=polyhedron.get_top_face().get_shapely()
            axes[i]=plot_utils.plot_shapely(top,axes[i],color="Black")
            
        # Interpolate power values on the grid
        z = griddata(receivers, powers, (x, y), method=interpolation_method, fill_value=np.min(powers))
        z_masked = np.ma.masked_where(np.logical_not(mask.reshape(x.shape)), z)
        # Plot the colormap
        axes[i].pcolormesh(x, y, z_masked, cmap=cmap, shading='auto', vmin=-120, vmax=0)
        #fig.colorbar(axes[i].pcolormesh(x, y, z_masked, cmap=cmap, shading='auto',vmin=-120,vmax=0), ax=axes[i], label='Power (dB)')
        # Plot the receiver points and their power values
        for receiver, power in zip(receivers, powers):
            axes[i].plot(receiver[0], receiver[1], 'ko', markersize=4)
            axes[i].text(receiver[0], receiver[1], f"{power:.1f}", fontsize=8)
        tx=txs[i]
        axes[i].plot(tx[0], tx[1], marker='X', color='lime', markersize=8)
        axes[i].text(tx[0], tx[1], "TX", fontsize=14,color="lime")
        
    fig.colorbar(axes[-1].pcolormesh(x, y, z_masked, cmap=cmap, shading='auto',vmin=-120,vmax=0), ax=axes[-1], label='Power (dB)')
    if save_name:
        plt.savefig(f"../results/plots/radiomap_{save_name}.eps", format='eps', dpi=300,bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return

def elevation_map():
    #hardcoded elevation map
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Elevation map',fontsize=20)
    cmap = plt.get_cmap('jet')
    
    pis=[-100,90]
    mid_pis=[-64,45]
    place=[36,-3.5]
    escalier=[62,-102]
    dinant=[35.3,-41]
    dinant_end=[-11,-94]
    chene=[73,41]
    data_points=[pis,mid_pis,place,escalier,dinant,dinant_end,chene]
    elevations=[16,21,27,34,29,29,24]

    street=Place_saint_jean.select_saint_jean_streets_hardcoded()
    #plot buildings
    place=Place_saint_jean.gen_place_saint_jean()
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
    z = griddata(data_points, elevations, (x, y), method="linear", fill_value=np.min(elevations))

    # Mask grid points outside the street polygon
    mask = np.array([street.contains(shapely.geometry.Point(x_, y_)) for x_, y_ in zip(x.ravel(), y.ravel())])
    z_masked = np.ma.masked_where(np.logical_not(mask.reshape(x.shape)), z)

    # Plot the colormap
    ax.pcolormesh(x, y, z_masked, cmap=cmap, shading='auto')
    fig.colorbar(ax.pcolormesh(x, y, z_masked, cmap=cmap, shading='auto'), ax=ax, label='Elevation (m)')
    plt.savefig("../results/plots/elevation_map.png", format='png', dpi=300,bbox_inches='tight')
    plt.show()


def run_saint_jean_simu(points_filename,save_name,order=2,flat=False,tx=None,tx_on_ground=True):
    place,tx,geometry=Place_saint_jean.create_place_saint_jean(points_filename,tx) if flat \
        else Place_saint_jean.create_slanted_place_saint_jean(points_filename,tx,tx_on_ground)
    #place_utils.plot_place(place, tx)
    save_name=f"{geometry}_{save_name}"
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,geometry=save_name,order=order)
    return tx,solved_em_path,solved_rays_path


def print_elapsed_time(start_time,end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {minutes} minutes {seconds} seconds\n")
    return


if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close("all")
    #elevation_map()
    def run_simu_suite(points_filename,tx,save_name="test",tx_on_ground=True):
        tx,solved_em_path,solved_rays_path=run_saint_jean_simu(points_filename,save_name=save_name,order=2,flat=False,tx=tx,tx_on_ground=tx_on_ground)
        #df=electromagnetism_utils.load_df(solved_em_path)
        # plt.close("all")
        # radiomap(df,tx,save_name,interpolation_method='linear')
        #electromagnetism_plots.plot_rx_rays_distribution(df,receivers=[1],save_name=save_name)
        #electromagnetism_plots.plot_rays_on_sphere_helper(df,receivers=[1],save_name)
        #problem=ray_tracing.RayTracingProblem.from_json(solved_rays_path)
        #problem.plot_specific_receiver(10)
        
        
    tx_in_middle=np.array([35,-6,0])
    tx_at_centraal=np.array([69,-54,0])
    
    tx_at_escalier=np.array([61,-93,0])
    tx_at_escalier+=np.array([-8,0,0])#on the side of the street
    
    tx_at_vieille_ble=np.array([73,40,0],dtype=np.float64)
    tx_at_vieille_ble+=np.array([-3,1.5,0]) #on the side of the street
    
    tx_at_pis=np.array([-102,92,0])
    tx_at_pis+=np.array([+6,0,0])#on the side of the street
     
    
    
    # start_time_rt = time.time()
    # points_filename="../data/place_saint_jean/place_saint_jean_160_points_d_4.json"
    # run_simu_suite(points_filename,tx=tx_at_centraal,save_name="tx_at_centraal_O2_12",tx_on_ground=False)
    # run_simu_suite(points_filename,tx=tx_in_middle,save_name="tx_in_middle_O2_12")
    # run_simu_suite(points_filename,tx=tx_at_escalier,save_name="tx_at_escalier_O2_12")
    # run_simu_suite(points_filename,tx=tx_at_vieille_ble,save_name="tx_at_vieille_ble_O2_12")
    # run_simu_suite(points_filename,tx=tx_at_pis,save_name="tx_at_pis_O2_12")
    
    # end_time_rt = time.time()
    # print_elapsed_time(start_time_rt,end_time_rt)
    
    #points_filename="../data/place_saint_jean/place_saint_jean_160_points_d_4.json"
    #tx,solved_em_path,solved_rays_path=run_saint_jean_simu(points_filename,save_name="tx_at_centraal_O2_12_25m",order=2,flat=False,tx=tx_at_centraal,tx_on_ground=False) 
   

    df1_path = "../results/plots/solved/data/slanted_place_saint_jean_tx_at_centraal_O2_12_160p_em_solved.csv"
    df1 = electromagnetism_utils.load_df(df1_path)
    name1="On_top_of_48m_building"
    df2_path = "../results/plots/solved/data/slanted_place_saint_jean_tx_in_middle_O2_12_160p_em_solved.csv"
    df2 = electromagnetism_utils.load_df(df2_path)
    name2="In_the_intersection"
    df3_path="../results/slanted_place_saint_jean_tx_at_centraal_O2_12_25m_160p_em_solved.csv"
    df3=electromagnetism_utils.load_df(df3_path)
    name3="On_top_of_25m_building"
    
    
    dfs=[df1,df2,df3]
    names=[name1,name2,name3]
    txs=[tx_at_centraal,tx_in_middle,tx_at_centraal]
    
    #electromagnetism_plots.compare_RMS_delay_spreads_cdf(dfs,names,save_name="RMS_delay_spreads_street_level_comparison")
    #electromagnetism_plots.compare_RMS_delay_spreads_pdf(dfs,names)
    
    #electromagnetism_plots.compare_angular_windows_cdf(dfs,names)
    electromagnetism_plots.compare_RMS_angular_spreads_cdf(dfs,names,"RMS_angular_spreads_street_level_comparison")
    
    #compare_radiomaps(dfs,txs)
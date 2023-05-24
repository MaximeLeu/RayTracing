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
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import patheffects
from scipy.interpolate import griddata
import shapely
import time
import geopandas as gpd

#self written imports
import electromagnetism_fun.electromagnetism_utils as electromagnetism_utils
import electromagnetism_fun.electromagnetism_plots as electromagnetism_plots
import electromagnetism_fun.materials_properties as material_properties
from electromagnetism_fun.multithread_solve import multithread_solve_place
import electromagnetism_fun.place_utils as place_utils
from place_utils import Place_saint_jean

import raytracing.plot_utils as plot_utils
import raytracing.file_utils as file_utils
plt.rcParams.update({'font.size': 20})

def compare_radiomaps(dfs, txs, save_name=None):
    interpolation_method = "linear"
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    txs = [txs] if not isinstance(txs, list) else txs

    fig = plt.figure(figsize=(15 * len(dfs), 15))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, len(dfs)),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15,
                     )

    place = Place_saint_jean.gen_place_saint_jean()
    street = Place_saint_jean.select_saint_jean_streets_hardcoded()
    grid_resolution = 300
    x, y = np.meshgrid(
        np.linspace(street.bounds[0], street.bounds[2], grid_resolution),
        np.linspace(street.bounds[1], street.bounds[3], grid_resolution)
    )
    mask = np.array([street.contains(shapely.geometry.Point(x_, y_)) for x_, y_ in zip(x.ravel(), y.ravel())])

    cmap = plt.get_cmap('jet')
    for i, (df, ax) in enumerate(zip(dfs, grid)):
        ax.set_title('Radiomap', fontsize=20)
        receivers = electromagnetism_utils.get_receiver_coordinates(dfs[i], two_dimensional=True)
        powers = electromagnetism_utils.get_power_db_each_receiver(dfs[i])

        for polyhedron in place.polyhedra:
            top = polyhedron.get_top_face().get_shapely()
            ax = plot_utils.plot_shapely(top, ax, color="Black")

        z = griddata(receivers, powers, (x, y), method=interpolation_method, fill_value=np.min(powers))
        z_masked = np.ma.masked_where(np.logical_not(mask.reshape(x.shape)), z)
        im = ax.pcolormesh(x, y, z_masked, cmap=cmap, shading='auto', vmin=-120, vmax=0)

        for receiver, power in zip(receivers, powers):
            ax.plot(receiver[0], receiver[1], 'ko', markersize=4)
            ax.text(receiver[0], receiver[1], f"{power:.1f}", fontsize=8)

        tx = txs[i]
        ax.plot(tx[0], tx[1], marker='X', color='lime', markersize=12,markeredgecolor="black")
        path_effect = [patheffects.withStroke(linewidth=4, foreground='black')] #to make the letter edge black
        text_obj = ax.text(tx[0], tx[1], "TX", fontsize=14, color="lime", weight="heavy")
        text_obj.set_path_effects(path_effect)

    grid[-1].cax.colorbar(im)
    grid[-1].cax.toggle_label(True)

    if save_name:
        plt.savefig(f"../results/plots/radiomap_{save_name}.png", format='png', dpi=300, bbox_inches='tight')
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

    tx_in_middle=np.array([35,-6,0])
    tx_at_centraal=np.array([69,-54,0])
    
    tx_at_escalier=np.array([61,-93,0])
    tx_at_escalier+=np.array([-8,0,0])#on the side of the street
    
    tx_at_vieille_ble=np.array([73,40,0],dtype=np.float64)
    tx_at_vieille_ble+=np.array([-3,1.5,0]) #on the side of the street
    
    tx_at_pis=np.array([-102,92,0])
    tx_at_pis+=np.array([+6,0,0])#on the side of the street
    
    def set_centraal_height(h):
        filename='../data/place_saint_jean/place_saint_jean.geojson'
        gdf = gpd.read_file(filename)    
        gdf.loc[gdf['id']=="way/224649053", "height"] = h
        gdf.to_file(filename, driver="GeoJSON")
    
    def create_all_plots():
        #change frequency in material_properties as well.
        #takes 2h to complete. (37min/simu)
        print(f"Solving at {material_properties.FREQUENCY/1e9} GHz")
        if material_properties.FREQUENCY==12.5*1e9:
            frequency="12G"
        elif material_properties.FREQUENCY==30*1e9:
            frequency="30G"
        start_time_rt = time.time()  
        points_filename="../data/place_saint_jean/place_saint_jean_160_points_d_4.json"
        set_centraal_height(h=48)
        tx_m,solved_em_path_m,solved_rays_path_m=run_saint_jean_simu(points_filename,save_name=f"tx_in_middle_O2_{frequency}",\
                                                               order=2,flat=False,tx=tx_in_middle,tx_on_ground=True)
        tx_c48,solved_em_path_c48,solved_rays_path_c48=run_saint_jean_simu(points_filename,save_name=f"tx_at_centraal_O2_{frequency}_48m",\
                                                               order=2,flat=False,tx=tx_at_centraal,tx_on_ground=False)
        set_centraal_height(h=30)
        tx_c30,solved_em_path_c30,solved_rays_path_c30=run_saint_jean_simu(points_filename,save_name=f"tx_at_centraal_O2_{frequency}_30m",\
                                                               order=2,flat=False,tx=tx_at_centraal,tx_on_ground=False)
        set_centraal_height(h=48)
            
        df_m = electromagnetism_utils.load_df(solved_em_path_m)    
        df_c48 = electromagnetism_utils.load_df(solved_em_path_c48)    
        df_c30 = electromagnetism_utils.load_df(solved_em_path_c30)    
        dfs=[df_m,df_c48,df_c30]
        names=['In_the_intersection','On_top_of_48m_building','On_top_of_30m_building']
        
        electromagnetism_plots.compare_RMS_delay_spreads_cdf(dfs,names,save_name=f"street_level_comparison_{frequency}")
        electromagnetism_plots.compare_RMS_angular_spreads_cdf(dfs,names,save_name=f"street_level_comparison_{frequency}")
        electromagnetism_plots.compare_angular_windows_cdf(dfs,names,save_name=f"street_level_comparison_{frequency}")
        
        compare_radiomaps([df_m,df_c48],[tx_in_middle,tx_at_centraal],save_name=f"tx_in_middle_and_48m_building_comparison_{frequency}")
        compare_radiomaps([df_c30],[tx_at_centraal],save_name=f'On_top_of_30m_building_{frequency}')
        
        electromagnetism_plots.compare_power_percent_coming_from_n_rays(df_m,nrays=[1,2,3,4],save_name=f"tx_in_middle_{frequency}")
        end_time_rt = time.time()
        print_elapsed_time(start_time_rt,end_time_rt)
        return
    
    create_all_plots()
    
    #df=electromagnetism_utils.load_df(solved_em_path)
    #elevation_map()
    #electromagnetism_plots.plot_rx_rays_distribution(df,receivers=[1],save_name=save_name)
    #electromagnetism_plots.plot_rays_on_sphere_helper(df,receivers=[1],save_name)
    #problem=ray_tracing.RayTracingProblem.from_json(solved_rays_path)
    #problem.plot_specific_receiver(10)
    #electromagnetism_plots.compare_angular_windows_cdf(dfs,names)
    #electromagnetism_plots.compare_RMS_delay_spreads_pdf(dfs,names)
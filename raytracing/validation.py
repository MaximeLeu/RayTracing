#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022
Code to validate the program
@author: Maxime Leurquin
"""
#packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#self written imports
from raytracing.multithread_solve import multithread_solve_place
from raytracing.materials_properties import FREQUENCY

from raytracing.electromagnetism import ElectromagneticField
from electromagnetism_plots import read_csv

import raytracing.ray_tracing as ray_tracing
import raytracing.electromagnetism_plots as electromagnetism_plots
import raytracing.place_utils as place_utils
import raytracing.electromagnetism_utils as electromagnetism_utils

plt.rcParams.update({'font.size': 20})


def get_corresponding_measures(month):
    files = {
        12.5*1e9: {"february": "claude_12_feb.csv", "october": "claude_12_oct.csv"},
        30*1e9: {"february": "claude_30_feb.csv", "october": "claude_30_oct.csv"}
    }
    if FREQUENCY not in files:
        raise ValueError(f"No measures for frequency {FREQUENCY}GHz")
    if month not in files[FREQUENCY]:
        raise ValueError(f"No measures for month {month}")
    return read_csv(files[FREQUENCY][month])


def read_simu(df,tx):
    """
    extracts x and y data from the dataframe
    put x and y in same format as the measurements
    """
    entrances,_,_,_=place_utils.levant_find_crucial_coordinates(filename="../data/place_levant_edited.geojson")
    maxwell_entrance=entrances[0]
    
    nreceivers=len(df['rx_id'].unique())
    simu_y=np.zeros(nreceivers)
    simu_x=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]#all data for this rx
        simu_y[receiver]=electromagnetism_utils.to_db(np.sum(rx_df["path_power"].values))
        rx_coord=rx_df["receiver"].values[0]
        #neglecting slope
        simu_x[receiver]=np.linalg.norm(maxwell_entrance[:-1] - rx_coord[:-1])
    return simu_x, simu_y


def get_path_loss(df,tx):
    nreceivers=len(df['rx_id'].unique())
    pl=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        d=np.linalg.norm(tx-rx_coord)
        pl[receiver]=ElectromagneticField.path_loss(d)
    return pl
    

def run_levant_simu(npoints=16,order=2,flat=False):
    place,tx,geometry=place_utils.create_flat_levant(npoints) if flat else place_utils.create_slanted_levant(npoints)
    #place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,geometry=geometry,order=order)
    return tx,solved_em_path,solved_rays_path
    

def plot_levant_vs_measures(tx,solved_em_path):
    df=electromagnetism_utils.load_df(solved_em_path)
    geometry=solved_em_path.split("/")[2]
    geometry=geometry.split("_")
    geometry="_".join(geometry[:3])
    print(f"PLotting {geometry} vs measures")
    
    pl=get_path_loss(df,tx)
    x,y=get_corresponding_measures("february")
    x1,y1=get_corresponding_measures("october")
    simu_x,simu_y=read_simu(df,tx)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz',fontsize=20)
    #ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,label='path loss')
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    
    ax.set_xlim([30, 83]) #for october measures
    
    ax.legend(fontsize=20)
    plt.show()
    plt.savefig(f"../results/plots/{geometry}_validation.eps", format='eps', dpi=300,bbox_inches='tight')
    return
    
    

def plot_slanted_vs_flat(npoints,order):
    """
    Run and plot a comparison of the simulation of the levant street on flat and slanted ground
    """  
    flat_tx,flat_solved_em_path,flat_solved_rays_path=run_levant_simu(npoints=npoints,order=order,flat=True)
    slanted_tx,slanted_solved_em_path,slanted_solved_rays_path=run_levant_simu(npoints=npoints,order=order,flat=False)
    
    slanted_df=electromagnetism_utils.load_df(slanted_solved_em_path)
    flat_df=electromagnetism_utils.load_df(flat_solved_em_path)
  
    #mes_x,mes_y=get_corresponding_measures("february")
    mes_x,mes_y=get_corresponding_measures("october")
    flat_x,flat_y=read_simu(flat_df,flat_tx)
    slanted_x,slanted_y=read_simu(slanted_df,slanted_tx)


    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz for flat and slanted ground',fontsize=20)
    
    #ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(flat_x,flat_y,color="blue",marker='o',label='flat ground')
    ax.plot(slanted_x,slanted_y,color="orange",marker='o',label='slanted ground')
    ax.plot(mes_x,mes_y,color='red', marker='o',label="october measures")
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    
    ax.set_xlim([30, 83]) #for october measures
    
    plt.savefig("../results/plots/flat_vs_slanted.eps", format='eps', dpi=300,bbox_inches='tight')
    plt.show()
    return
    

def plot_sensitivity_tx(movements,npoints,order,plot_name):
    """
    Compares results when the tx antenna was moved slightly
    movements is a list of ndarrays.
    """
    assert isinstance(movements,list),f"got movements={movements} but expected list of nparrays"
    normal_tx,solved_em_path,_=run_levant_simu(npoints=npoints,order=order,flat=False)
    normal_df=electromagnetism_utils.load_df(solved_em_path)
    normal_y=electromagnetism_utils.get_power_db_each_receiver(normal_df)    

    fig, ax = plt.subplots(figsize=(20, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(movements)))
    for idx, movement in enumerate(movements):
        place,tx,geometry=place_utils.create_slanted_levant(npoints)
        tx[0]=tx[0]+movement
        solved_em_path2,_= multithread_solve_place(place=place,tx=tx,geometry=geometry,
                                                                 order=order,save_name=f"{geometry}_{npoints}p_{plot_name}")
        moved_df = electromagnetism_utils.load_df(solved_em_path2)
        moved_y = electromagnetism_utils.get_power_db_each_receiver(moved_df)
        ax.plot(range(npoints), moved_y,color=colors[idx], marker='o', label=f'tx moved of +{movement}m')
    
    ax.plot(range(npoints),normal_y,color="black",marker='o',label='baseline')
    ax.set(title=f'TX sensitivity study at {FREQUENCY/(1e9)} GHz',
        xticks=range(0,npoints),
        xlabel='Receiver #',
        ylabel='Received power [dB]')
    ax.legend(fontsize=20)
    ax.grid()
    plt.savefig(f"../results/plots/{plot_name}.eps", format='eps', dpi=300,bbox_inches='tight')
    plt.show()
    return
    
    

    


#Restart the kernel before launching if using IPython
if __name__ == '__main__':
    # file_utils.chdir_to_file_dir(__file__)
    # plt.close('all')
    
    tx,solved_em_path,solved_rays_path=run_levant_simu(npoints=16*5,order=2,flat=False)
    plot_levant_vs_measures(tx[0],solved_em_path)
    #electromagnetism_plots.plot_order_importance(solved_em_path)
    #electromagnetism_plots.EM_fields_plots(solved_em_path,name="slanted_final")
    
    problem=ray_tracing.RayTracingProblem.from_json(solved_rays_path)
    problem.plot_specific_receiver(30)

    
    
    def run_levant_sensitivity():
        x_movements=[np.array([-0.3,0,0]),
                     np.array([-0.2,0,0]),
                     np.array([-0.1,0,0]),
                     np.array([0.1,0,0]),
                     np.array([0.2,0,0]),
                     np.array([0.3,0,0])
                     ]
        
        y_movements=[np.array([0,-0.3,0]),
                     np.array([0,-0.2,0]),
                     np.array([0,-0.1,0]),
                     np.array([0,0.1,0]),
                     np.array([0,0.2,0]),
                     np.array([0,0.3,0])
                     ]
        
        z_movements=[np.array([0,0,-0.3]),
                     np.array([0,0,-0.2]),
                     np.array([0,0,-0.1]),
                     np.array([0,0,0.1]),
                     np.array([0,0,0.2]),
                     np.array([0,0,0.3])
                     ]
        
        plot_sensitivity_tx(x_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_x")
        plot_sensitivity_tx(y_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_y")
        plot_sensitivity_tx(z_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_z")
        return
    #plot_slanted_vs_flat(npoints=16*5,order=2)
   
    
    

  
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
import scipy as sc

#self written imports
from electromagnetism_fun.multithread_solve import multithread_solve_place
from electromagnetism_fun.materials_properties import FREQUENCY
from electromagnetism_fun.electromagnetism import ElectromagneticField
import electromagnetism_fun.electromagnetism_plots as electromagnetism_plots
import electromagnetism_fun.electromagnetism_utils as electromagnetism_utils
import electromagnetism_fun.place_utils as place_utils
from place_utils import Place_du_levant
import raytracing.ray_tracing as ray_tracing


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
    return electromagnetism_plots.read_csv(files[FREQUENCY][month])


def read_simu(df,tx):
    """
    extracts x and y data from the dataframe
    put x and y in same format as the measurements
    """
    entrances,_,_,_=Place_du_levant.levant_find_crucial_coordinates(filename="../data/place_levant_edited.geojson")
    maxwell_entrance=entrances[0]
    
    powers=electromagnetism_utils.get_power_db_each_receiver(df)
    rx_coords=np.array(electromagnetism_utils.get_receiver_coordinates(df))
    simu_x = np.linalg.norm(maxwell_entrance[:-1] - rx_coords[:, :-1], axis=1) #neglecting slope
    return simu_x, powers


def get_path_loss(df, tx):
    rx_coords = np.array(electromagnetism_utils.get_receiver_coordinates(df))
    distances = np.linalg.norm(tx - rx_coords, axis=1)
    pl = ElectromagneticField.path_loss(distances) / 2.6 #gain to match the measures
    return pl
    

def run_levant_simu(npoints=16,order=2,flat=False):
    place,tx,geometry=Place_du_levant.create_flat_levant(npoints) if flat else Place_du_levant.create_slanted_levant(npoints)
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
    #x,y=get_corresponding_measures("february")
    x1,y1=get_corresponding_measures("october")
    simu_x,simu_y=read_simu(df,tx)
    
    #xlim the simu to match the measures
    simu_x=simu_x[simu_x<=np.max(x1)]
    cut=len(simu_x)
    simu_y=simu_y[:cut]
    pl=pl[:cut]
    
    f_mes = sc.interpolate.interp1d(x1, y1,fill_value='extrapolate')
    simu_interp_meas=f_mes(simu_x)
    simu_RMSE=electromagnetism_utils.RMSE(simu_y,simu_interp_meas)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    #ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label=f'simulation. RMSE={simu_RMSE:.2f}')
    ax.plot(simu_x,pl,label='path loss')

    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz',fontsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20,loc='lower left'), ax.grid()
    plt.savefig(f"../results/plots/levant/{geometry}_validation.eps", format='eps', dpi=300,bbox_inches='tight')
    plt.show()
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
    
    #xlim the simu to match the measures
    flat_x=flat_x[flat_x<np.max(mes_x)]
    cut=len(flat_x)
    flat_y=flat_y[:cut]
    
    slanted_x=slanted_x[slanted_x<np.max(mes_x)]
    cut=len(slanted_x)
    slanted_y=slanted_y[:cut]
    
    #interpolate the measures on the simulation points:
    f_mes = sc.interpolate.interp1d(mes_x, mes_y,fill_value='extrapolate')
    flat_interp_meas=f_mes(flat_x)
    slanted_interp_meas=f_mes(slanted_x)
    flat_RMSE=electromagnetism_utils.RMSE(flat_y,flat_interp_meas)
    slanted_RMSE=electromagnetism_utils.RMSE(slanted_y,slanted_interp_meas)
    

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(flat_x,flat_y,color="blue",marker='o',label=f'flat ground. RMSE={flat_RMSE:.2f}')
    ax.plot(slanted_x,slanted_y,color="orange",marker='o',label=f'slanted ground. RMSE={slanted_RMSE:.2f}')
    ax.plot(mes_x,mes_y,color='red', marker='o',label="october measures")
    
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz for flat and slanted ground',fontsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20,loc='lower left'),ax.grid()
    
    plt.savefig("../results/plots/levant/flat_vs_slanted.eps", format='eps', dpi=300,bbox_inches='tight')
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

    rmse=np.zeros(len(movements))
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(movements)))
    for idx, movement in enumerate(movements):
        place,tx,geometry=Place_du_levant.create_slanted_levant(npoints)
        tx[0]=tx[0]+movement
        solved_em_path2,_= multithread_solve_place(place=place,tx=tx,geometry=geometry,
                                                                 order=order,save_name=f"{geometry}_{npoints}p_{plot_name}")
        moved_df = electromagnetism_utils.load_df(solved_em_path2)
        moved_y = electromagnetism_utils.get_power_db_each_receiver(moved_df)
        rmse[idx]=electromagnetism_utils.RMSE(normal_y,moved_y)
        ax.plot(range(npoints), moved_y,color=colors[idx], marker='o', label=f'tx+={movement}m RMSE={rmse[idx]:.2f}')
        
    
    ax.plot(range(npoints),normal_y,color="black",marker='o',label='baseline')
    ax.set(title=f'TX sensitivity study at {FREQUENCY/(1e9)} GHz',
        xticks=range(0,npoints),
        xlabel='Receiver #',
        ylabel='Received power [dB]')
    ax.legend(fontsize=20,loc='lower left'),ax.grid()
    plt.savefig(f"../results/plots/levant/{plot_name}.eps", format='eps', dpi=300,bbox_inches='tight')
    plt.show()
    print(f"RMSE for each movement: {rmse}")
    return
    
    

    


#Restart the kernel before launching if using IPython
if __name__ == '__main__':
    # file_utils.chdir_to_file_dir(__file__)
    # plt.close('all')

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
        #close the created plot to start computing of the next one.
        #plot_sensitivity_tx(x_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_x")
        #plot_sensitivity_tx(y_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_y")
        plot_sensitivity_tx(z_movements,npoints=16*1,order=2,plot_name="levant_sensitivity_z")
        return
    
    def make_all_plots():
        #make sure TX gain=1 and RX gain=0.5
        #make sure antennas are not isotropic
        tx,solved_em_path,solved_rays_path=run_levant_simu(npoints=16*1,order=2,flat=False) 
        plot_levant_vs_measures(tx[0],solved_em_path)
        #electromagnetism_plots.plot_order_importance(solved_em_path)
        #electromagnetism_plots.EM_fields_plots(solved_em_path,name="slanted_final")
        # problem=ray_tracing.RayTracingProblem.from_json(solved_rays_path)
        # problem.plot_specific_receiver(30)
        #plot_slanted_vs_flat(npoints=16*5,order=2)    
        
        
    make_all_plots()    
    #run_levant_sensitivity()
    
   
    
    

  
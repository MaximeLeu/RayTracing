#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022
Code to validate the program
@author: Maxime Leurquin
"""
#packages
import csv
import numpy as np
import matplotlib.pyplot as plt

#self written imports
from raytracing.electromagnetism_utils import to_db
from raytracing.multithread_solve import multithread_solve_place
from raytracing.materials_properties import FREQUENCY

from raytracing.electromagnetism import ElectromagneticField

import raytracing.place_utils as place_utils
import raytracing.file_utils as file_utils
import raytracing.electromagnetism_utils as electromagnetism_utils


def read_csv(file):
    """
    Measures from C. Oestges and D. Vanhoenacker-Janvier,
    Experimental validation and system applications of ray-tracing model in built-up areas,
    Electronics Letters, vol. 36, no. 5, p. 461, 2000.
    were extracted and stored in csv using https://apps.automeris.io/wpd/
    this reads the measures from the csv
    """
    x = []
    y = []
    with open(f'../results/validation_measures/{file}','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            x.append(round(float(row[0])))
            y.append(float(row[1]))
    return x,y


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
    nreceivers=len(df['rx_id'].unique())
    simu_y=np.zeros(nreceivers)
    simu_x=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]#all data for this rx
        simu_y[receiver]=to_db(np.sum(rx_df["path_power"].values))
        rx_coord=rx_df["receiver"].values[0]
        #assuming tx is on the border of the maxwell building, neglecting slope
        simu_x[receiver]=np.linalg.norm(tx[:-1] - rx_coord[:-1])#dist maxwell base-rx
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
    place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,geometry=geometry,order=order)
    return tx,solved_em_path,solved_rays_path
    
def run_small_simu(npoints=16,order=2):
    place,tx,geometry=place_utils.create_small_place(npoints)
    place_utils.plot_place(place, tx)
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
    
    #plotting
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz',fontsize=20)
    ax.plot(x,y,color='green', marker='o',label="february measures")
    #ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,label='path loss')
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    plt.savefig(f"../results/plots/{geometry}_validation.eps", format='eps', dpi=1000,bbox_inches='tight')
    return
    

def plot_slanted_vs_flat(solved_slanted_path,solved_flat_path):
    """
    plots a comparison of the simulation of the levant street on flat and slanted ground
    the simulations must have been run before with 
    -save_name='slanted_levant_simu'
    -save_name='flat_levant_simu'

    """
    slanted_df=electromagnetism_utils.load_df(solved_slanted_path)
    slanted_tx=np.array([46.76, 3.77, 12.44])
    
    flat_df=electromagnetism_utils.load_df(solved_flat_path)
    flat_tx=np.array([46.76, 3.77,9.44])
    
    mes_x,mes_y=get_corresponding_measures("february")
    flat_x,flat_y=read_simu(flat_df,flat_tx)
    slanted_x,slanted_y=read_simu(slanted_df,slanted_tx)
    
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz for flat and slanted ground',fontsize=20)
    
    #ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(flat_x,flat_y,color="orange",marker='o',label='flat ground')
    ax.plot(slanted_x,slanted_y,color="blue",marker='o',label='slanted ground')
    ax.plot(mes_x,mes_y,color='green', marker='o',label="february measures")
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    plt.savefig("../results/plots/flat_vs_slanted.eps", format='eps', dpi=1000,bbox_inches='tight')
    
    
    
#TODO FIX TX movement and recompute
def plot_sensitivity_tx(normal_path,moved_path):
    """
    Compares results when the tx antenna was moved slightly
    """
    normal_df=electromagnetism_utils.load_df(normal_path)
    normal_tx=np.array([46.76, 3.77, 12.44])
    
    moved_df=electromagnetism_utils.load_df(moved_path)
    moved_tx=np.array([46.76, 3.77,9.44])
    
    mes_x,mes_y=get_corresponding_measures("february")
    normal_x,normal_y=read_simu(normal_df,normal_tx)
    moved_x,moved_y=read_simu(moved_df,moved_tx)
    
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'TX sensitivity study at {FREQUENCY/(1e9)} GHz',fontsize=20)
    ax.plot(normal_x,normal_y,color="orange",marker='o',label='baseline')
    ax.plot(moved_x,moved_y,color="blue",marker='o',label='moved')
    ax.plot(mes_x,mes_y,color='green', marker='o',label="february measures")
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    plt.savefig("../results/plots/tx_sensitivity.eps", format='eps', dpi=1000,bbox_inches='tight')
    
    
def plot_small_vs_path_loss(tx,solved_em_path):
    """
    comparison of the fields obtained on the small place and path loss.
    """
    df=electromagnetism_utils.load_df(solved_em_path)
    npoints=len(df['rx_id'].unique())
    
    simu_y=np.zeros(npoints)
    simu_x=np.zeros(npoints)
    pl=np.zeros(npoints)
    for receiver in range(npoints):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=rx_df["receiver"].values[0]
        d=np.linalg.norm(tx-rx_coord)
        pl[receiver]=ElectromagneticField.path_loss(d)
        simu_x[receiver]=d #distance TX-RX
        simu_y[receiver]=to_db(np.sum(rx_df["path_power"].values))

    #plots
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between path loss and model at {FREQUENCY/1e9} GHz', fontsize=20)
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,marker='o',label='path loss')
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('RX-TX distance',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    #plt.savefig("../results/plots/small_place_vs_pathLoss.eps", format='eps', dpi=1000,bbox_inches='tight')
    return
    

def plot_measures_only():
    #plt.rcParams.update({'font.size': 22})
    x,y=read_csv("claude_12_feb.csv")
    x1,y1=read_csv("claude_12_oct.csv")
    
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Measurements at 12.5 GHz')
    ax.plot(x,y,color='green', marker='o',label="february")
    ax.plot(x1,y1,color='red', marker='o',label="october")
    ax.grid()
    ax.set_xlabel('distance from Maxwell building [m]')
    ax.set_ylabel('Received power [dB]')
    ax.set_xlim(30,90)
    ax.set_ylim(-60,-35)
    ax.legend(loc='lower left')
    plt.savefig("../results/validation_measures/claude_125.eps", dpi=150,bbox_inches='tight')
    
    x,y=read_csv("claude_30_feb.csv")
    x1,y1=read_csv("claude_30_oct.csv")
    fig2 = plt.figure(figsize=(20,8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title('Measurements at 30 GHz')
    ax2.plot(x,y,color='green', marker='o',label="february")
    ax2.plot(x1,y1,color='red', marker='o',label="october")
    ax2.grid()
    ax2.set_xlabel('distance from Maxwell building [m]')
    ax2.set_ylabel('Received power [dB]')
    ax2.set_xlim(30,90)
    ax2.set_ylim(-60,-35)
    ax2.legend(loc='lower left')
    plt.savefig("../results/validation_measures/claude_30.eps", dpi=150,bbox_inches='tight')
    plt.show()
    return



#ALWAYS RESTART KERNEL BEFORE LAUNCH
if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close('all')
    #plot_claude_only()
    
    #tx,solved_em_path,solved_rays_path=run_small_simu(npoints=16,order=2)
    #plot_small_vs_path_loss(tx,solved_em_path)
    
    tx,solved_em_path,solved_rays_path=run_levant_simu(npoints=3,order=2,flat=False)
    plot_levant_vs_measures(tx[0],solved_em_path)
    
    #save_name=f'{geometry}_{npoints}p'
    solved_slanted_path="../results/{save_name}_ray_solved.json"
    solved_flat_path="../results/{save_name}_ray_solved.json"
    #plot_slanted_vs_flat()
    

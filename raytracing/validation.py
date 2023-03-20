#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022
Code to validate the program
@author: max
"""
#packages
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

#self written imports
from raytracing import file_utils
from electromagnetism import to_db,path_loss
import place_utils
import raytracing.geometry as geom

from multithread_solve import multithread_solve_place
from materials_properties import FREQUENCY

from theoretical_validation import two_rays_fields


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


def plot_measures_simu_comparison(df,maxwell_base,tx,compare_two_rays):
    """
    df: dataframe of the solved problem
    """
    def read_simu(df,maxwell_base):
        """
        extracts x and y data from the dataframe
        put x and y in same format as claude
        """
        nreceivers=len(df['rx_id'].unique())
        simu_y=np.zeros(nreceivers)
        simu_x=np.zeros(nreceivers)
        for receiver in range(nreceivers):
            rx_df=df.loc[df['rx_id'] == receiver]#all data for this rx
            simu_y[receiver]=to_db(np.sum(rx_df["path_power"].values))
            rx_coord=rx_df["receiver"].values[0]
            #Measures the distance between the maxwell and the receiver (neglecting slope)
            simu_x[receiver]=np.linalg.norm(maxwell_base[:-1] - rx_coord[:-1])#dist maxwell-rx
        return simu_x, simu_y

    #path loss
    nreceivers=len(df['rx_id'].unique())
    pl=np.zeros(nreceivers)
    sol_two_rays=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        d=np.linalg.norm(tx-rx_coord)
        pl[receiver]=path_loss(d)
        tx_base=maxwell_base #assuming tx_base at same height of rx base
        rx_base=rx_coord
        L=np.linalg.norm(tx_base-rx_base)
        sol_two_rays[receiver]=two_rays_fields(L,ztx=tx[2],zrx=rx_coord[2],slope=-3.8)

    #claude's measures
    if FREQUENCY==12.5*1e9:
        x,y=read_csv("claude_12_feb.csv")
        x1,y1=read_csv("claude_12_oct.csv")
    elif FREQUENCY==30*1e9:
        x,y=read_csv("claude_30_feb.csv")
        x1,y1=read_csv("claude_30_oct.csv")
    else:
        assert 1==2, "frequency does not match claude's"

    #my simulation
    simu_x,simu_y=read_simu(df,maxwell_base)

    #plotting
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/(1e9)} GHz',fontsize=20)
    ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,label='path loss')
    
    if compare_two_rays:
        ax.plot(simu_x,sol_two_rays,color="black",marker='o',label='two rays')
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('distance to Maxwell building base [m]',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    plt.savefig("../plots/levant_validation.eps", format='eps', dpi=1000,bbox_inches='tight')
    return


def small_vs_path_loss(npoints=15,order=2):
    """
    comparison of the fields obtained on the small place and path loss.
    """
    place,tx,_=place_utils.create_small_place(npoints)
    place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='small',order=order)
    df=file_utils.load_df(solved_em_path)

    simu_y=np.zeros(npoints)
    simu_x=np.zeros(npoints)
    pl=np.zeros(npoints)
    for receiver in range(npoints):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=rx_df["receiver"].values[0]
        d=np.linalg.norm(tx-rx_coord)
        pl[receiver]=path_loss(d)
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
    #plt.savefig("../plots/small_place_vs_pathLoss.eps", format='eps', dpi=1000,bbox_inches='tight')
    return

def levant_vs_measures(npoints=15,order=2,compare_two_rays=False):
    #place,tx,_=place_utils.create_levant_place(npoints)
    place,tx,_=place_utils.create_slanted_levant(npoints=npoints)
    place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='levant_claude',order=order)
    df=file_utils.load_df(solved_em_path)
    tx=tx[0]
    ground=place.surface.polygons[2]
    maxwell_base=geom.polygon_line_intersection(ground,[tx,tx+np.array([0,0,100])])
    maxwell_base+=np.array([0,0,1.2]) #adding rx heigth
    plot_measures_simu_comparison(df,maxwell_base,tx,compare_two_rays)
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
    plt.savefig("../plots/claude_125.eps", dpi=150,bbox_inches='tight')
    
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
    plt.savefig("../plots/claude_30.eps", dpi=150,bbox_inches='tight')
    plt.show()
    return

#ALWAYS RESTART KERNEL BEFORE LAUNCH
if __name__ == '__main__':
    plt.close('all')
    #plot_claude_only()
    start_time = time.time()
    levant_vs_measures(npoints=15*1,order=2,compare_two_rays=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("")
    print(f"Elapsed time: {minutes} minutes {seconds} seconds")
    #small_vs_path_loss(npoints=16*3,order=2)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022
Code to validate the program
@author: max
"""
#self written imports
from raytracing import file_utils
from electromagnetism import to_db
from electromagnetism import RX_GAIN,TX_GAIN
import place_utils

from multithread_solve import multithread_solve_place
from materials_properties import FREQUENCY
#packages
import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
import csv

LAMBDA=c/FREQUENCY

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


def compute_path_loss(tx,rx):
    d=np.linalg.norm(tx-rx)
    pr_pt=RX_GAIN*TX_GAIN*(LAMBDA/(4*pi*d))**2
    pl=10*np.log10(pr_pt)
    return pl


def plot_claude_comparison(df,maxwell_base,tx):
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
            simu_y[receiver]=to_db(np.sum(rx_df["path_power"].values)/FREQUENCY)
            #simu_y[receiver]=FieldPower.compute_power(rx_df["field_strength"].values,STYLE=2)        
            rx_coord=rx_df["receiver"].values[0]
            dist_maxwell=np.linalg.norm(maxwell_base-rx_coord) #distance between the receiver and the maxwell
            simu_x[receiver]=dist_maxwell       
        return simu_x, simu_y
    
    #path loss
    nreceivers=len(df['rx_id'].unique())
    pl=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        pl[receiver]=compute_path_loss(tx, rx_coord)
    
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
    
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/1e9} GHz')
    ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,marker='o',label='path loss')
    ax.grid()  
    ax.set_xlabel('distance to Maxwell building [m]')
    ax.set_ylabel('Received power [dB]')
    ax.legend()
    plt.show() 
    return


def small_vs_path_loss(npoints=15,order=2):
    """
    comparison of the fields obtained on the small place and path loss.
    """
    place,tx,geometry=place_utils.create_small_place(npoints)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='small',order=order)
    df=file_utils.load_df(solved_em_path)
    
    simu_y=np.zeros(npoints)
    simu_x=np.zeros(npoints)
    pl=np.zeros(npoints)
    for receiver in range(npoints):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=rx_df["receiver"].values[0]
        pl[receiver]=compute_path_loss(tx, rx_coord)
        simu_x[receiver]=np.linalg.norm(rx_coord-tx) #distance TX-RX
        simu_y[receiver]=to_db(np.sum(rx_df["path_power"].values)/FREQUENCY)
        #simu_y[receiver]=FieldPower.compute_power(rx_df["field_strength"].values,STYLE=2)
    
    
    #plots
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between path loss and model at {FREQUENCY/1e9} GHz')
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,marker='o',label='path loss')
    ax.grid()  
    ax.set_xlabel('RX-TX distance')
    ax.set_ylabel('Received power [dB]') 
    ax.legend()
    plt.show() 
    return

def levant_vs_measures(npoints=15,order=3):
    place,tx,geometry=place_utils.create_levant_place(npoints)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='levant_claude',order=order) 
    df=file_utils.load_df(solved_em_path)
    tx=tx[0]
    maxwell_base=[tx[0],tx[1],1]
    plot_claude_comparison(df,maxwell_base,tx)
    return



#ALWAYS RESTART KERNEL BEFORE LAUNCH
if __name__ == '__main__':
    #care to go modify the E field frequency adequately in materials properties as well beforehand.
    plt.close('all')

    levant_vs_measures(npoints=16*3,order=2)
    #small_vs_path_loss(npoints=16,order=2)
    
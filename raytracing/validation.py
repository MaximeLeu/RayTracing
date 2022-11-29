#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022
Code to validate the program
@author: max
"""
#self written imports
from raytracing import file_utils
from electromagnetism import FieldPower
from electromagnetism import RX_GAIN,TX_GAIN
import place_utils

from multithread_solve import multithread_solve_place
from materials_properties import FREQUENCY
#packages
import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
import csv



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



def plot_claude_comparison(df,maxwell_base):
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
            simu_y[receiver]=FieldPower.compute_power(rx_df["field_strength"].values,STYLE=2)        
            rx_coord=rx_df["receiver"].values[0]
            dist_maxwell=np.linalg.norm(maxwell_base-rx_coord) #distance between the receiver and the maxwell
            simu_x[receiver]=dist_maxwell       
        return simu_x, simu_y
    
    def path_loss(df,maxwell_base):
        nreceivers=len(df['rx_id'].unique())
        d=np.zeros(nreceivers)
        for receiver in range(nreceivers):
            rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
            d[receiver]=np.linalg.norm(rx_coord-maxwell_base)
        pr_pt=RX_GAIN*TX_GAIN*(c/(4*pi*d*FREQUENCY))**2 #pr/pt
        pl=10*np.log10(pr_pt)
        return pl
    
    if FREQUENCY==12.5*1e9:
        x,y=read_csv("claude_12_feb.csv")
        x1,y1=read_csv("claude_12_oct.csv")
    elif FREQUENCY==30*1e9:
        x,y=read_csv("claude_30_feb.csv")
        x1,y1=read_csv("claude_30_oct.csv")
    else:
        assert 1==2, "frequency does not match claude's"

    simu_x,simu_y=read_simu(df,maxwell_base)    
    
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {FREQUENCY/1e9} GHz')
    ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,path_loss(df),marker='o',label='path loss')
    ax.grid()  
    ax.set_xlabel('distance to Maxwell building [m]')
    ax.set_ylabel('Received power [dB]')
    ax.legend()
    plt.show() 
    return


def small_vs_path_loss(npoints=15):
    """
    comparison of the fields obtained on the small place and path loss.
    """
    place,tx,geometry=place_utils.create_small_place(npoints)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='small')
    df=file_utils.load_df(solved_em_path)
    
    simu_y=np.zeros(npoints)
    simu_x=np.zeros(npoints)
    path_loss=np.zeros(npoints)
    for receiver in range(npoints):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=rx_df["receiver"].values[0]
        d=np.linalg.norm(rx_coord-tx) #distance TX-RX
        simu_x[receiver]=d
        pr_pt=RX_GAIN*TX_GAIN*(c/(4*pi*d*FREQUENCY))**2 #pr/pt
        path_loss[receiver]=10*np.log10(pr_pt)
        simu_y[receiver]=FieldPower.compute_power(rx_df["field_strength"].values,STYLE=2)
    
    
    #plots
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between path loss and model at {FREQUENCY/1e9} GHz')
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,path_loss,marker='o',label='path loss')
    ax.grid()  
    ax.set_xlabel('RX-TX distance')
    ax.set_ylabel('Received power [dB]') 
    ax.legend()
    plt.show() 
    return

def claude_comparison(npoints=15):
    place,tx,geometry=place_utils.create_levant_place(npoints)
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='levant_claude') 
    df=file_utils.load_df(solved_em_path)
    maxwell_base=[tx[0],tx[1],1]
    plot_claude_comparison(df,maxwell_base)
    return



#ALWAYS RESTART KERNEL BEFORE LAUNCH
if __name__ == '__main__':
    #care to go modify the E field frequency adequately in materials properties as well beforehand.
    plt.close('all')
    
    #claude_comparison(npoints=15)
    small_vs_path_loss(npoints=15)
    
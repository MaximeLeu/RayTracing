#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:38:52 2022

@author: max
"""


#self written imports
from ray_tracing import RayTracingProblem
import raytracing.geometry as geom
from raytracing import plot_utils,file_utils
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data,vm_to_db

#packages
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 22})
import pandas as pd
import csv



MAXWELL_HEIGHT_CLAUDE=40
CLAUDE_N_POINTS=10
TX_CLAUDE = np.array([110,-40,MAXWELL_HEIGHT_CLAUDE+1.2]).reshape(1, 3)
RX_HEIGHT=1.2


ST_BARBE_COORD=np.array([-15,-46,RX_HEIGHT])
MAXWELL_COORDINATES=np.array([80,-10,0])
DIST=MAXWELL_COORDINATES[0]-ST_BARBE_COORD[0]
print(f"MAXWELL: {MAXWELL_COORDINATES} barb: {ST_BARBE_COORD} dist: {DIST}" )



def plot_place_claude():  
    
    #useful to get Maxwell's coordinates
    geometry_filename='../data/place_levant.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    plot_utils.add_points_to_3d_ax(ax=ax, points=TX_CLAUDE, label="TX",marker='o')
    
   
    rx= ST_BARBE_COORD
    step=DIST/CLAUDE_N_POINTS
    for receiver in range(CLAUDE_N_POINTS):
        rx =rx+np.array([step,0,0])
        rx=rx.reshape(-1, 3)
        plot_utils.add_points_to_3d_ax(ax=ax, points=rx, label=f"RX{receiver}",marker='+')
        
        
    place.center_3d_plot(ax)   
    ax = place.plot3d(ax=ax)
    plt.show(block=False)
    plt.pause(0.001) 

def read_csv(file):
    #using this https://apps.automeris.io/wpd/
    x = []
    y = []
    with open(f'../results/validation_measures/{file}','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            x.append(round(float(row[0])))
            y.append(float(row[1]))
    return x,y 

def claude_comparison(df,freq):
    if freq==12.5:
        #12.5 GHz
        x,y=read_csv("claude_12_feb.csv")
        x1,y1=read_csv("claude_12_oct.csv")
    else:
        #30GHz
        x,y=read_csv("claude_30_feb.csv")
        x1,y1=read_csv("claude_30_oct.csv")
    

    nreceivers=len(df['rx_id'].unique())
    simu_y=np.zeros(nreceivers)
    simu_x=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        simu_y[receiver]=vm_to_db(np.sum(rx_df['path_power'].values))
        
        rx_coord=(df.loc[df['rx_id'] == receiver]['receiver'].values[0]) #this is a string
        rx_coord=rx_coord[1:-1] #remove brackets
        rx_coord=np.fromstring(rx_coord, sep=',')
        
        dist_maxwell=np.linalg.norm(MAXWELL_COORDINATES-rx_coord) #distance between the receiver and the maxwell
        simu_x[receiver]=dist_maxwell
       
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Comparison between measurements and simulation at {freq} GHz')
    ax.plot(x,y,color='green', marker='o',label="february measures")
    ax.plot(x1,y1,color='red', marker='o',label="october measures")
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.grid()  
    ax.legend()
    plt.show() 

def mani_comparison(df):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    
    GHZ=3.8
    x=np.arange(1,16,1)
    mani_y=np.array([-89,-83,-80,-79,-80,-82,-83,-91,-98,-90,-93,-90,-95,-95,-104])
   
    nreceivers=len(df['rx_id'].unique())
    simu_y=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        simu_y[receiver]=vm_to_db(np.sum(rx_df['path_power'].values))
    
    ax.plot(x,simu_y,color='green', marker='o',label="simulated results")
    ax.plot(x, mani_y,color='blue', marker='o',label="mani measurements")
      
    ax.set_title(f'Comparison between measurements and simulation at {GHZ} GHz')
    ax.set_ylabel('Received power [dB]') 
    ax.set_xlabel("measurement positions")
    ax.set_xlim(1,15)
    ax.grid()  
    ax.legend()
    plt.show()
    return

if __name__ == '__main__':
    plt.close('all')
    
    geometry='small'
    
    df=pd.read_csv("../results/small.csv")
    plot_place_claude()
    claude_comparison(df,12.5)
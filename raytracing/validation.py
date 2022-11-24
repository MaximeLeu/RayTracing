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
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data,to_db,harvested_power,compute_power
from electromagnetism import RX_GAIN,TX_GAIN

from multithread_solve import multithread_solve_place
from materials_properties import FREQUENCY
#packages
import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 22})
import pandas as pd
import csv


MAXWELL_HEIGHT=40 #find it in the geojson

RX_HEIGHT=1.2
ST_BARBE_COORD=np.array([-15,-46,RX_HEIGHT])
MAXWELL_COORDINATES=np.array([80,-10,0])
DIST=MAXWELL_COORDINATES[0]-ST_BARBE_COORD[0]



def create_place_claude(tx,npoints):
    #create and show the place
    print(f"MAXWELL: {MAXWELL_COORDINATES} barb: {ST_BARBE_COORD} dist: {DIST}" )
    
    geometry_filename='../data/place_levant.geojson'
    geom.preprocess_geojson(geometry_filename)
    place = geom.generate_place_from_rooftops_file(geometry_filename)
    fig = plt.figure("the place")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX",marker='o')
    
    rx= ST_BARBE_COORD
    step=DIST/npoints
    for receiver in range(npoints):
        rx =rx+np.array([step,0,0])
        rx=rx.reshape(-1, 3)
        plot_utils.add_points_to_3d_ax(ax=ax, points=rx, label=f"RX{receiver}",marker='+')
        place.add_set_of_points(rx)
            
    place.center_3d_plot(ax)   
    ax = place.plot3d(ax=ax)
    plt.show(block=False)
    plt.pause(0.001) 
    return place
    


def plot_claude_comparison(df):
    """
    df: dataframe of the solved problem for the given frequency
    frequency: either 12.5 or 30 GHz
    """
    #read claude's measures from csv
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

    def read_simu():
        """
        extracts x and y data from the dataframe
        put x and y in same format as claude
        """
        nreceivers=len(df['rx_id'].unique())
        simu_y=np.zeros(nreceivers)
        simu_x=np.zeros(nreceivers)
        for receiver in range(nreceivers):
            rx_df=df.loc[df['rx_id'] == receiver]
            simu_y[receiver]=compute_power(rx_df["field_strength"].values,STYLE=3)        
            rx_coord=(df.loc[df['rx_id'] == receiver]['receiver'].values[0])        
            dist_maxwell=np.linalg.norm(MAXWELL_COORDINATES-rx_coord) #distance between the receiver and the maxwell
            simu_x[receiver]=dist_maxwell       
        return simu_x, simu_y
    
    def path_loss(df):
        nreceivers=len(df['rx_id'].unique())
        d=np.zeros(nreceivers)
        for receiver in range(0,nreceivers):
            rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
            d[receiver]=np.linalg.norm(rx_coord-MAXWELL_COORDINATES)
        print(f"distance between maxwell and each RX {d}m")
        pr_pt=RX_GAIN*TX_GAIN*(c/(4*pi*d*FREQUENCY))**2 #pr/pt
        pl=10*np.log10(pr_pt)
        return pl
    
    if FREQUENCY==12.5*1e9:
        #12.5 GHz
        x,y=read_csv("claude_12_feb.csv")
        x1,y1=read_csv("claude_12_oct.csv")
    elif FREQUENCY==30*1e9:
        #30GHz
        x,y=read_csv("claude_30_feb.csv")
        x1,y1=read_csv("claude_30_oct.csv")
    else:
        assert 1==2, "frequency does not match claude's"

    simu_x,simu_y=read_simu()    
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
# def plot_mani_comparison(df):
#     fig = plt.figure(figsize=(20,20))
#     ax = fig.add_subplot(1, 1, 1)
    
#     GHZ=3.8
#     x=np.arange(1,16,1)
#     mani_y=np.array([-89,-83,-80,-79,-80,-82,-83,-91,-98,-90,-93,-90,-95,-95,-104])
   
#     nreceivers=len(df['rx_id'].unique())
#     simu_y=np.zeros(nreceivers)
#     for receiver in range(nreceivers):
#         rx_df=df.loc[df['rx_id'] == receiver]
#         #TODO correct
#         simu_y[receiver]=to_db(np.sum(rx_df['path_power'].values))
    
#     ax.plot(x,simu_y,color='green', marker='o',label="simulated results")
#     ax.plot(x, mani_y,color='blue', marker='o',label="mani measurements")
      
#     ax.set_title(f'Comparison between measurements and simulation at {GHZ} GHz')
#     ax.set_ylabel('Received power [dB]') 
#     ax.set_xlabel("measurement positions")
#     ax.set_xlim(1,15)
#     ax.grid()  
#     ax.legend()
#     plt.show()
#     return

def compare_claude_driver(solveRays=False):#,solveEM=True):  
    TX_CLAUDE = np.array([110,-40,MAXWELL_HEIGHT+1.2]).reshape(1, 3)
    place_claude=create_place_claude(TX_CLAUDE,npoints=15)
    solved_em_path="../results/levant_claude_em_solved.csv"
    solved_rays_path="../results/levant_claude_ray_solved.json"
    
    if solveRays==True:
        #solve rays and em fields
        solved_em_path,solved_rays_path= multithread_solve_place(place=place_claude,tx=TX_CLAUDE,save_name='levant_claude') 
        
    # if solveEM==True:
    #     #does not work because problem.load not working
    #     #only solve em fields
    #     problem=RayTracingProblem(None,None)
    #     problem.load(solved_rays_path)
    #     solved_em_path=my_field_computation(problem,solved_em_path)
    
    df=file_utils.load_df(solved_em_path)
    plot_claude_comparison(df)
    return

def compare_small_PL_driver():
    def create_place():
        geometry_filename='../data/small.geojson'  
        geom.preprocess_geojson(geometry_filename)
        place = geom.generate_place_from_rooftops_file(geometry_filename)
        # 2. Create TX and RX
        ground_center = place.get_centroid()
        tx = ground_center + [-50, 5, 1]
        tx = tx.reshape(-1, 3)  
        fig = plt.figure("the place")
        fig.set_dpi(300)
        ax = fig.add_subplot(projection='3d')
        plot_utils.add_points_to_3d_ax(ax=ax, points=tx, label="TX",marker='o')
        npoints=15
        step=4
        rx=ground_center-[30,-10,0]
        for receiver in range(npoints):
            rx =rx+np.array([step,0,0])
            rx=rx.reshape(-1, 3)
            plot_utils.add_points_to_3d_ax(ax=ax, points=rx, label=f"RX{receiver}",marker='+')
            place.add_set_of_points(rx)
        place.center_3d_plot(ax)   
        ax = place.plot3d(ax=ax)
        plt.show(block=False)
        plt.pause(0.001) 
        return place, tx
    
    place,tx=create_place()
    solved_em_path,solved_rays_path= multithread_solve_place(place=place,tx=tx,save_name='small')
    df=file_utils.load_df(solved_em_path)
    
    
    nreceivers=len(df['rx_id'].unique())
    simu_y=np.zeros(nreceivers)
    simu_x=np.zeros(nreceivers)
    path_loss=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=(df.loc[df['rx_id'] == receiver]['receiver'].values[0])        
        d=np.linalg.norm(rx_coord-tx) #distance TX-RX
        simu_x[receiver]=d
        pr_pt=RX_GAIN*TX_GAIN*(c/(4*pi*d*FREQUENCY))**2 #pr/pt
        path_loss[receiver]=10*np.log10(pr_pt)
        simu_y[receiver]=compute_power(rx_df["field_strength"].values,STYLE=3)
    
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


#ALWAYS RESTART KERNEL BEFORE LAUNCH
if __name__ == '__main__':
    #care to go modify the E field frequency adequately in materials properties as well beforehand.
    plt.close('all')
    
    #compare_claude_driver(solveRays=False)
    
    compare_small_PL_driver()
    
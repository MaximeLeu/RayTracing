#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:29:01 2023

@author: max
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors


import raytracing.electromagnetism_utils as electromagnetism_utils
import raytracing.file_utils as file_utils
import raytracing.plot_utils as plot_utils

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


def plot_measures_only():
    #plt.rcParams.update({'font.size': 22})
    x,y=read_csv("claude_12_feb.csv")
    x1,y1=read_csv("claude_12_oct.csv")
    
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,color='green', marker='o',label="february")
    ax.plot(x1,y1,color='red', marker='o',label="october")
    ax.set(title='Measurements at 12.5 GHz',
       xlabel='distance from Maxwell building [m]',ylabel='Received power [dB]',
       xlim=(30, 90),ylim=(-60, -35))
    ax.grid()
    ax.legend(loc='lower left')
    plt.savefig("../results/validation_measures/claude_125.eps", dpi=150,bbox_inches='tight')
    
    x,y=read_csv("claude_30_feb.csv")
    x1,y1=read_csv("claude_30_oct.csv")
    fig2 = plt.figure(figsize=(20,8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x,y,color='green', marker='o',label="february")
    ax2.plot(x1,y1,color='red', marker='o',label="october")
    ax2.set(title='Measurements at 30 GHz',
        xlabel='distance from Maxwell building [m]',ylabel='Received power [dB]',
        xlim=(30, 90),ylim=(-60, -35))
    ax2.grid()
    ax2.legend(loc='lower left')
    fig.savefig("../results/validation_measures/claude_30.eps", dpi=150,bbox_inches='tight')
    plt.show()
    return




def EM_fields_plots(df_path, order=3, name="unnamed_plot"):
    """
    Plots for each receiver, the power contribution of each path type and the power delay profile.
    Generates one pdf for each 50 receivers. 
    """
    df = electromagnetism_utils.load_df(df_path)
    df_list = electromagnetism_utils.split_df(df, [])
    
    for k, current_df in enumerate(df_list):
        receivers=current_df['rx_id'].unique()
        n_receivers = len(receivers)
        
        fig, axes = plt.subplots(n_receivers, 2, figsize=(16, 5 * n_receivers), dpi=75)
        fig.subplots_adjust(hspace=.5)

        for i, (ax1, ax2) in enumerate(axes):
            receiver=receivers[i]
            rx_df = current_df[current_df['rx_id'] == receiver]
            path_types = rx_df['path_type'].unique()
            width = 0.35
            
            for path_type in path_types:
                data_for_type = rx_df[rx_df['path_type'] == path_type]
                color_for_type, position, ticks = plot_utils.set_color_for_type(path_type, order)

                individual_powers = [electromagnetism_utils.to_db(power) for power in data_for_type['path_power']]
                total_power = electromagnetism_utils.to_db(np.sum(data_for_type['path_power']))

                ax1.bar(x=position, height=total_power, width=width, color=color_for_type, label=path_type)
                ax2.stem(data_for_type['time_to_receiver'].values * (1e9), individual_powers,
                         linefmt=color_for_type, label=path_type, basefmt=" ")
                
            ax1.set(title=f'Total power from sources RX{receiver}',
                    xticks=range(len(ticks)), xticklabels=ticks,
                    ylabel='Received power [dB]')
            ax1.grid()
            ax2.set(title=f'Power delay Profile RX{receiver}',
                    xlabel='time to reach receiver (ns)',
                    ylabel='Received power [dB]')
            ax2.legend()
            ax2.grid()

        fig.savefig(f"../results/plots/EM_plots_{name}{k}.pdf", dpi=100, bbox_inches='tight')
        plt.close(fig)
    return




def get_receiver_data(df,receivers):
    rx_matrix_list=[]
    tx_matrix_list=[]
    for receiver in receivers:
        data=df.loc[df["rx_id"]==receiver]
        
        tx_angles=data['tx_angles'].to_numpy()
        tx_angles=np.deg2rad(np.vstack(tx_angles))
        rx_angles=data['rx_angles'].to_numpy()
        rx_angles=np.deg2rad(np.vstack(rx_angles))
        power=data['path_power'].to_numpy()
        
        tx_path_data = np.column_stack([tx_angles[:,0], tx_angles[:,1], power])
        rx_path_data = np.column_stack([rx_angles[:,0], rx_angles[:,1], power])
        
        rx_matrix_list.append(rx_path_data)
        tx_matrix_list.append(tx_path_data)
    
    stacked_tx_matrix = np.vstack(tx_matrix_list)
    stacked_rx_matrix = np.vstack(rx_matrix_list)

    return stacked_tx_matrix,stacked_rx_matrix
    
    

def plot_data_on_sphere(ax, data, title):
    x, y, z = electromagnetism_utils.to_cartesian(r=1, theta=data[:, 0], phi=data[:, 1])

    # Create a sphere
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1)

    # Plot the data on a 3D sphere
    power=electromagnetism_utils.to_db(data[:, 2])
    scatter = ax.scatter(x, y, z, c=power, cmap='jet')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Received power [dB]', fontsize=12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax=plot_utils.ensure_axis_orthonormal(ax)

    # Add axis lines through center of sphere
    center = np.array([0, 0, 0])
    ax.plot([center[0], 1], [center[1], 0], [center[2], 0], color='r')
    ax.plot([center[0], 0], [center[1], 1], [center[2], 0], color='g')
    ax.plot([center[0], 0], [center[1], 0], [center[2], 1], color='b')
    ax.text(1.1, 0, 0, 'X', color='r', fontsize=12)
    ax.text(0, 1.1, 0, 'Y', color='g', fontsize=12)
    ax.text(0, 0, 1.1, 'Z', color='b', fontsize=12)

    ax.set_title(title, fontsize="14")
    
    
def plot_rays_on_sphere(data_tx, data_rx):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Ray Distribution: Outgoing Rays from the TX Antenna and Incoming Rays at the RX Antenna")
    ax1 = fig.add_subplot(121, projection='3d')
    plot_data_on_sphere(ax1, data_tx, "Distribution of outgoing ray from the tx antenna")

    ax2 = fig.add_subplot(122, projection='3d')
    plot_data_on_sphere(ax2, data_rx, "Distribution of incoming rays at the rx antenna")

    plt.show()
    return    


    
def plot_order_importance(solved_em_path):
    """
    Successively plots the power: 
    -only if first order paths are taken into account
    -only if first and second order paths are taken into account
    -only if first, second and third order paths are taken into account
    -etc.
    """
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(1, 1, 1)
    
    df=electromagnetism_utils.load_df(solved_em_path)
    nreceivers=len(df['rx_id'].unique())
    x=np.arange(0,nreceivers,1)
    max_order=electromagnetism_utils.find_df_order(df)
    for order in range(0,max_order+1):
        order_df=electromagnetism_utils.get_data_up_to_order(df,order)
        y=electromagnetism_utils.get_power_db_each_receiver(order_df)
        ax.plot(x,y,marker='o',label=f"up to order {order}")
    ax.grid()
    ax.legend()
    ax.set(title='Impact of simulating higher orders',
            xlabel='Receiver #',
            xticks=range(0,nreceivers),
            ylabel='Received power [dB]')
    plt.plot()
    return




if __name__=='__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close('all')
    df_path="../results/slanted_levant_16p_em_solved.csv"
    df=electromagnetism_utils.load_df(df_path)
    
    
    plot_order_importance(df_path)
    plot_measures_only()
    # nreceivers=len(df['rx_id'].unique())
    # receivers=np.arange(0,nreceivers)
    
    # data_tx,data_rx=get_receiver_data(df,receivers)
    # print(data_tx)
    # plot_rays_on_sphere(data_tx,data_rx)
    
    
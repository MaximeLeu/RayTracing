#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:29:01 2023

@author: max
"""
import matplotlib.pyplot as plt
import numpy as np

import raytracing.electromagnetism_utils as electromagnetism_utils
import raytracing.file_utils as file_utils
import raytracing.plot_utils as plot_utils

file_utils.chdir_to_file_dir(__file__)

def EM_fields_plots(df_path,order=3,name="unnamed_plot"):
    df=electromagnetism_utils.load_df(df_path)
    nreceivers=len(df['rx_id'].unique())
    nrows=nreceivers
    ncols=2

    if nreceivers>16*6:
        print("Too many receivers; can't display EM field plots")
        return

    fig = plt.figure("EM fields data",figsize=(16,5*nrows))
    fig.set_dpi(150)
    fig.subplots_adjust(hspace=.5)

    i=1
    for receiver in range(0,nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        path_types=rx_df['path_type'].unique()
        ax1 = fig.add_subplot(nrows, ncols,i)
        ax2 = fig.add_subplot(nrows, ncols,i+1)
        i+=2
        width = 0.35
        for path_type in path_types:
            data_for_type=rx_df.loc[rx_df['path_type'] == path_type]
            color_for_type,position,ticks=plot_utils.set_color_for_type(path_type,order)
            #total power from each source
            power=electromagnetism_utils.to_db(np.sum(data_for_type["path_power"].values))
            ax1.bar(x=position, height=power,width=width,color=color_for_type,label=path_type)
            #power delay profile
            nelem=len(data_for_type['field_strength'])
            individual_powers=np.zeros(nelem)
            for j in range(nelem):
                individual_powers[j]=electromagnetism_utils.to_db(data_for_type['path_power'].values[j])
            ax2.stem(data_for_type['time_to_receiver'].values*(1e9), individual_powers,linefmt=color_for_type,label=path_type,basefmt=" ")

        ax1.set_title(f'Total power from sources RX{receiver}')
        ax1.set_xticks(range(0,len(ticks)), ticks)
        ax1.set_ylabel('Received power [dB]')
        ax1.grid()

        ax2.set_title(f'Power delay Profile RX{receiver}')
        ax2.set_xlabel('time to reach receiver (ns)')
        ax2.set_ylabel('Received power [dB]')
        ax2.legend()
        ax2.grid()
        plt.savefig(f"../results/plots/EM_plots_{name}'.pdf", dpi=300,bbox_inches='tight')
        #plt.show()

    return fig

    
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
    cbar.set_label('Power', fontsize=12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #ensure orthogonality
    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')
    ax.axis('equal')

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


if __name__=='__main__':
    plt.close('all')
    df_path="../results/flat_levant_simu_em_solved.csv"
    df=electromagnetism_utils.load_df(df_path)
    
    nreceivers=len(df['rx_id'].unique())
    receivers=np.arange(0,nreceivers)
    
    data_tx,data_rx=get_receiver_data(df,receivers)
    print(data_tx)
    plot_rays_on_sphere(data_tx,data_rx)
    
    
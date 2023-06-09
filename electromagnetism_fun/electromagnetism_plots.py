#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:29:01 2023

@author: Maxime Leurquin
"""
import csv
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import electromagnetism_fun.electromagnetism_utils as electromagnetism_utils
import raytracing.file_utils as file_utils
import raytracing.plot_utils as plot_utils

plt.rcParams.update({'font.size': 18})


def read_csv(file):
    """
    Measures from C. Oestges and D. Vanhoenacker-Janvier,
    Experimental validation and system applications of ray-tracing model in built-up areas,
    Electronics Letters, vol. 36, no. 5, p. 461, 2000.
    were extracted and stored in csv using https://apps.automeris.io/wpd/
    this reads the measures from the csv
    """
    x, y = [], []
    with open(f'../results/validation_measures/{file}', 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            x.append(round(float(row[0])))
            y.append(float(row[1]))
    return x, y


def plot_measures_only():
    #plt.rcParams.update({'font.size': 22})
    x, y = read_csv("claude_12_feb.csv")
    x1, y1 = read_csv("claude_12_oct.csv")

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(x, y, color='green', marker='o', label="february")
    ax.plot(x1, y1, color='red', marker='o', label="october")
    ax.set(title='Measurements at 12.5 GHz',
            xlabel='distance from Maxwell building [m]', ylabel='Received power [dB]',
            xlim=(30, 90), ylim=(-60, -35))
    ax.grid()
    ax.legend(loc='lower left')
    plt.savefig("../results/validation_measures/claude_125.eps",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    x, y = read_csv("claude_30_feb.csv")
    x1, y1 = read_csv("claude_30_oct.csv")
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(x, y, color='green', marker='o', label="february")
    ax.plot(x1, y1, color='red', marker='o', label="october")
    ax.set(title='Measurements at 30 GHz',
            xlabel='distance from Maxwell building [m]', ylabel='Received power [dB]',
            xlim=(30, 90), ylim=(-60, -35))
    ax.grid()
    ax.legend(loc='lower left')
    fig.savefig("../results/validation_measures/claude_30.eps",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("saved figures in /results/validation_measures")
    return



def set_color_for_type(path_type, order):
    types = electromagnetism_utils.generate_path_types_up_to_order(order)
    ind = types.index(path_type)
    colors = cm.rainbow(np.linspace(0, 1, len(types)))
    color = mcolors.rgb2hex(colors[ind])
    return color, ind, types


def EM_fields_plots(df_path, name="unnamed_plot"):
    """
    Plots for each receiver, the power contribution of each path type and the power delay profile.
    Generates one pdf file for each 50 receivers. 
    """
    df = electromagnetism_utils.load_df(df_path)
    df_list = electromagnetism_utils.split_df(df, [])
    order = electromagnetism_utils.find_df_order(df)

    for k, current_df in enumerate(df_list):
        receivers = current_df['rx_id'].unique()
        n_receivers = len(receivers)

        fig, axes = plt.subplots(
            n_receivers, 2, figsize=(16, 5 * n_receivers), dpi=75)
        fig.subplots_adjust(hspace=.5)

        for i, (ax1, ax2) in enumerate(axes):
            receiver = receivers[i]
            rx_df = current_df[current_df['rx_id'] == receiver]
            path_types = rx_df['path_type'].unique()
            path_types_reversed = path_types[::-1]  # for clearer plotting

            for path_type in path_types_reversed:
                data_for_type = rx_df[rx_df['path_type'] == path_type]
                color_for_type, position, ticks = set_color_for_type(
                    path_type, order)

                individual_powers = [electromagnetism_utils.to_db(
                    power) for power in data_for_type['path_power']]
                total_power = electromagnetism_utils.to_db(
                    np.sum(data_for_type['path_power']))

                ax1.bar(x=position, height=total_power, width=0.35,
                        color=color_for_type, label=path_type)
                ax2.stem(data_for_type['time_to_receiver'].values * (1e9), individual_powers,
                         linefmt=color_for_type, label=path_type, basefmt=" ")
            ax1.set(title=f'Total power from sources RX{receiver}',
                    xticks=range(len(ticks)), xticklabels=ticks,
                    ylabel='Received power [dB]')
            
            ax2.set(title=f'Power delay Profile RX{receiver}',
                    xlabel='time to reach receiver (ns)',
                    ylabel='Received power [dB]')
            ax2.legend(fontsize=10)
            ax1.grid(),ax2.grid()

        fig.savefig(
            f"../results/plots/EM_plots_{name}{k}.pdf", dpi=150, bbox_inches='tight')
        print(f"saved figure in /results/plots/{name}")
        plt.close(fig)
    return


def get_receiver_data(df, receivers):
    """
    given a receiver and a solved electromagnetic dataframe returns 
    all the tx and rx angles and the power for the receiver.
    """
    rx_matrix_list = []
    tx_matrix_list = []
    for receiver in receivers:
        data = df.loc[df["rx_id"] == receiver]
        if data["path_type"].values[0] == "Impossible":
            continue
        tx_angles = data['tx_angles'].to_numpy()
        tx_angles = np.deg2rad(np.vstack(tx_angles))
        rx_angles = data['rx_angles'].to_numpy()
        rx_angles = np.deg2rad(np.vstack(rx_angles))
        power = data['path_power'].to_numpy()

        tx_path_data = np.column_stack(
            [tx_angles[:, 0], tx_angles[:, 1], power])
        rx_path_data = np.column_stack(
            [rx_angles[:, 0], rx_angles[:, 1], power])

        rx_matrix_list.append(rx_path_data)
        tx_matrix_list.append(tx_path_data)

    stacked_tx_matrix = np.vstack(tx_matrix_list)
    stacked_rx_matrix = np.vstack(rx_matrix_list)

    return stacked_tx_matrix, stacked_rx_matrix


def plot_rx_rays_distribution(df, receivers, save_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # data_tx[:0]=elevation, data_tx[:1]= azimuth. Same for data_rx
    data_tx, data_rx = get_receiver_data(df, receivers)
    for receiver in receivers:
        power = electromagnetism_utils.to_db(data_rx[:, 2])
        data_rx[:, 1] = electromagnetism_utils.bring_angles_deg_in_0_360_range(
            np.degrees(data_rx[:, 1]))
        data_rx[:, 0] = electromagnetism_utils.bring_angles_deg_in_0_360_range(
            np.degrees(data_rx[:, 0]))

        data_tx[:, 1] = electromagnetism_utils.bring_angles_deg_in_0_360_range(
            np.degrees(data_tx[:, 1]))
        data_tx[:, 0] = electromagnetism_utils.bring_angles_deg_in_0_360_range(
            np.degrees(data_tx[:, 0]))

        scatter1 = ax1.scatter(
            data_rx[:, 1], data_rx[:, 0], c=power, cmap='jet')
        ax2.scatter(data_tx[:, 1], data_tx[:, 0], c=power, cmap='jet')

    cbar = plt.colorbar(
        scatter1, ax=[ax1, ax2], shrink=0.5, orientation='horizontal', pad=0.15)
    cbar.set_label('Power [dB]', fontsize=14)


    ax1.set(title="RX angles distribution",
            ylabel="theta (degrees)",
            xlabel='phi (degrees)')
    ax2.set(title="TX angles distribution",
            ylabel="theta (degrees)",
            xlabel='phi (degrees)')    
    ax1.grid(),ax2.grid()
    plt.savefig(
        f"../results/plots/rays_distribution_{save_name}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    #plt.show()
    return


def plot_data_on_sphere(ax, data, title):
    x, y, z = electromagnetism_utils.to_cartesian(
        r=1, theta=data[:, 0], phi=data[:, 1])

    # Create a sphere
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1)

    # Plot the data on a 3D sphere
    power = electromagnetism_utils.to_db(data[:, 2])
    scatter = ax.scatter(x, y, z, c=power, cmap='jet')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5,
                        orientation='horizontal', pad=0.02)
    cbar.set_label('Received power [dB]', fontsize=12)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
           xlabel='X', ylabel='Y', zlabel='Z')

    ax = plot_utils.ensure_axis_orthonormal(ax)

    # Add axis lines through center of sphere
    center = np.array([0, 0, 0])
    ax.plot([center[0], 1], [center[1], 0], [center[2], 0], color='r')
    ax.plot([center[0], 0], [center[1], 1], [center[2], 0], color='g')
    ax.plot([center[0], 0], [center[1], 0], [center[2], 1], color='b')
    ax.text(1.1, 0, 0, 'X', color='r', fontsize=12)
    ax.text(0, 1.1, 0, 'Y', color='g', fontsize=12)
    ax.text(0, 0, 1.1, 'Z', color='b', fontsize=12)

    ax.set_title(title, fontsize="14")
    return


def plot_rays_on_sphere(df, receivers, save_name=None):
    data_tx, data_rx = get_receiver_data(df, receivers)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), subplot_kw=dict(projection='3d'))
    plot_data_on_sphere(axes[0], data_tx, "Distribution of outgoing ray from the tx antenna")
    plot_data_on_sphere(axes[1], data_rx, "Distribution of incoming rays at the rx antenna")
    if save_name:
        plt.savefig(f"../results/plots/rays_on_sphere_{save_name}.eps", format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
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
    fig, ax = plt.subplots(figsize=(20, 8))
    df = electromagnetism_utils.load_df(solved_em_path)
    nreceivers = len(df['rx_id'].unique())
    x = np.arange(nreceivers)
    max_order = electromagnetism_utils.find_df_order(df)
    for order in range(max_order+1):
        order_df = electromagnetism_utils.get_data_up_to_order(df, order)
        y = electromagnetism_utils.get_power_db_each_receiver(order_df)
        linestyle = '--' if order == 3 else ':' if order == 4 else '-'
        ax.plot(x, y, marker='o', linestyle=linestyle,
                label=f"up to order {order}")
    
    ax.set(title='Impact of simulating higher orders',
           xlabel='Receiver #',
           xticks=range(nreceivers),
           ylabel='Received power [dB]')
    ax.grid(),ax.legend()
    plt.savefig("../results/plots/order_importance.eps",
                format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    return

def compute_angular_windows(df,POWER_PERCENT=0.99):
    #POWER_PERCENT=percent of power that must be in the cluster
    
    #This function finds the closest bundle of rays that contribute to 70% of the total received power.
    #Then it computes the spread between the two furthest apart rays of the bundle.
    #Assuming power_percent=70%
    # METHOD1:
        #When one ray is known to be in the bundle, because he contributes to more than 30% of the total power.
        # 1:Pick the highest power contributor: P
        # 2:Add power contributors geographically closest to P until 70% of total power is gathered
        # 3:Compute spread.

    # METHOD2:
        #When there is no clear power contributor.
        #Apply method 1 to every power contributor
        #See which spread is the smallest.
      
    # DISTANCE:
        # 1:put all [elevation,azimuth] values on a sphere of radius 1.
        # 2:Convert to euclidian coordinates
        # 3:Compute euclidian distance.
   
  
    def find_power_contributors_cluster(data,centroid,other_points):
        #given a centroid, and an array of points around it (including itself), find the smallest cluster of points
        #meeting the power percentage contribution.
        #return the indexes of all the point of the cluster.
        total_power = data["path_power"].sum()
        #distance between the centroid and every other point:
        distances = np.sqrt(np.sum((other_points - centroid)**2, axis=1))
        contributors=[]
        gathered_power=0
        for _ in distances:
            if gathered_power/total_power>=POWER_PERCENT:
                return contributors
            closest=np.argmin(distances)
            contributors.append(closest)
            distances[closest]=np.inf
            gathered_power+=data.loc[closest,"path_power"]
        return contributors
    
    def find_two_furthest_points_in_array(points):
        #return the indexes of the two furtest apart points of the array
        #points should be given in cartesian coordinates
        if len(points)==2:
            return [0,1]
        furthest_distance=0
        most_separated_points_idx=[]
        for i,point in enumerate(points):
            distances = np.sqrt(np.sum((points - point)**2, axis=1))
            furthest_point_idx=np.argmax(distances)
            max_dist=np.max(distances)
            if max_dist>furthest_distance:
                furthest_distance=max_dist
                most_separated_points_idx=[i,furthest_point_idx]
        return most_separated_points_idx
    
    
    df = df.dropna(subset=['tx_angles', 'rx_angles'])
    receivers = df['rx_id'].unique()
    az_spreads, el_spreads = np.zeros(len(receivers)), np.zeros(len(receivers))
    for i, receiver in enumerate(receivers):
        data = df.loc[df["rx_id"] == receiver].copy().reset_index()
        total_power = data["path_power"].sum()
        rx_el = data["rx_angles"].apply(lambda x: x[0])
        rx_az = data["rx_angles"].apply(lambda x: x[1])
        cartesian = np.column_stack(electromagnetism_utils.to_cartesian(r=1, theta=rx_el, phi=rx_az))

        max_power_contributor_idx = data["path_power"].idxmax()
        max_contributor_percent=data.loc[max_power_contributor_idx,"path_power"]/total_power
        
        if max_contributor_percent>POWER_PERCENT:
            #one single ray is enough to meet power target
            best_el_spread, best_az_spread=0, 0 
        elif max_contributor_percent>(1-POWER_PERCENT):
            #METHOD 1:the max contributor must be in the cluster to reach the power target  
            max_contributor_coords=cartesian[max_power_contributor_idx]
            contributors_idx=find_power_contributors_cluster(data,centroid=max_contributor_coords,other_points=cartesian)
            #compute spread:
            most_separated_points_idx=find_two_furthest_points_in_array(cartesian[contributors_idx])
            el1,el2=rx_el[most_separated_points_idx]
            az1,az2=rx_az[most_separated_points_idx]
            best_el_spread=electromagnetism_utils.angle_distance(el1,el2)
            best_az_spread=electromagnetism_utils.angle_distance(az1,az2)
        else:
            #METHOD 2: cluster is not trivial
            min_spread=np.inf
            for power_contributor in cartesian:
                contributors_idx=find_power_contributors_cluster(data,centroid=power_contributor,other_points=cartesian)
                #compute spread.
                most_separated_points_idx=find_two_furthest_points_in_array(cartesian[contributors_idx])              
                el1,el2=rx_el[most_separated_points_idx]
                az1,az2=rx_az[most_separated_points_idx]
                el_spread=electromagnetism_utils.angle_distance(el1,el2)
                az_spread=electromagnetism_utils.angle_distance(az1,az2)
                if az_spread*el_spread<min_spread:
                    best_el_spread, best_az_spread= el_spread, az_spread
            
        el_spreads[i]=best_el_spread
        az_spreads[i]=best_az_spread
    return az_spreads,el_spreads

def compute_RMS_delay_spreads(df):
    #https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1407-6-201706-I!!PDF-E.pdf
    df = df.dropna(subset=['tx_angles', 'rx_angles'])
    receivers = df['rx_id'].unique()
    trms=np.zeros(len(receivers))
    for i, receiver in enumerate(receivers):
        data = df.loc[df["rx_id"] == receiver]
        total_power = data["path_power"].sum()
        t0 = data['time_to_receiver'].min()
        t_mean=np.dot(data["time_to_receiver"],data["path_power"])/total_power-t0
        trms[i]=np.sqrt(np.dot((data['time_to_receiver']-t0-t_mean)**2,data["path_power"])/total_power)
    return trms


def compute_RMS_angular_spreads(df):
    #https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1407-6-201706-I!!PDF-E.pdf
    df = df.dropna(subset=['tx_angles', 'rx_angles'])
    receivers = df['rx_id'].unique()
    RMS_az_spreads, RMS_el_spreads = np.zeros(len(receivers)), np.zeros(len(receivers))
    for i, receiver in enumerate(receivers):
        data = df.loc[df["rx_id"] == receiver].copy().reset_index()
        total_power = data["path_power"].sum()
        
        most_powerful_ray=data["path_power"].idxmax()
        power_el=data.loc[most_powerful_ray,"rx_angles"][0]
        power_az=data.loc[most_powerful_ray,"rx_angles"][1]
        
        #put the angles relative to the most powerful ray
        rx_el = data["rx_angles"].apply(lambda x: electromagnetism_utils.angle_distance(x[0],power_el))
        rx_az = data["rx_angles"].apply(lambda x: electromagnetism_utils.angle_distance(x[1],power_az))
        
        rx_el=np.deg2rad(rx_el)
        rx_az=np.deg2rad(rx_az)
        
        el_mean=np.dot(rx_el,data["path_power"])/total_power
        az_mean=np.dot(rx_az,data["path_power"])/total_power
        
        RMS_el_spreads[i]=np.sqrt(np.dot((rx_el-el_mean)**2,data["path_power"])/total_power)
        RMS_az_spreads[i]=np.sqrt(np.dot((rx_az-az_mean)**2,data["path_power"])/total_power)
        
    RMS_el_spreads=np.degrees(RMS_el_spreads)
    RMS_az_spreads=np.degrees(RMS_az_spreads)
    return RMS_az_spreads,RMS_el_spreads


def dataframe_drop_NLOS(dataframe):
    df=dataframe.copy()
    df= df.dropna(subset=['tx_angles', 'rx_angles'])
    receivers = df['rx_id'].unique()
    for i, receiver in enumerate(receivers):
        data = df.loc[df["rx_id"] == receiver].copy()
        if "LOS" not in data["path_type"].unique():
            # drop entire "data" dataframe from df.
            df.drop(data.index, inplace=True)
    return df
    
def compute_percent_coming_from_n_rays(df,n=1):
    #For each receiver, compute the percentage of the received power coming
    #from the n most powerful rays
    #df = df.dropna(subset=['tx_angles', 'rx_angles'])
    df=dataframe_drop_NLOS(df)
    receivers = df['rx_id'].unique()
    percents = np.zeros(len(receivers))
    for i, receiver in enumerate(receivers):
        data = df.loc[df["rx_id"] == receiver].copy()
        total_power = data["path_power"].sum()
        powers=np.sort(np.array(data["path_power"]))
        if len(powers)>=n:
            percents[i]=sum(powers[-n:])/total_power
        else:
            percents[i]=sum(powers)/total_power
    return percents*100


def compare_RMS_delay_spreads_pdf(dfs,names,save_name=None):
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    names = [names] if not isinstance(names, list) else names
    fig, axes = plt.subplots(nrows=1,ncols=len(names),figsize=(10*len(names), 6))
    axes=[axes] if not isinstance(axes,np.ndarray) else axes
    
    for i, df in enumerate(dfs):
        trms = compute_RMS_delay_spreads(df) * 1e9
        sns.histplot(trms, kde=False, stat='density', bins=30, ax=axes[i])
        axes[i].set(xlabel='RMS delay Spread (ns)',
               ylabel='Probability Density',
               title=f"Delay spread probability density {names[i]}")
        axes[i].grid()
    if save_name:
        plt.savefig(f"../results/plots/delay_spread_pdf_{save_name}.eps",
                format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return

    
def compare_RMS_delay_spreads_cdf(dfs,names,save_name=None):
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    names = [names] if not isinstance(names, list) else names
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'r', 'g', 'm']

    for i, df in enumerate(dfs):
        trms = compute_RMS_delay_spreads(df) * 1e9
        x, y = electromagnetism_utils.cdf(trms)
        ax.plot(x, y, '-' + colors[i], label=names[i])

    ax.set(xlabel="RMS delay spread (ns)",
           ylabel='Cumulative probability',
           title="RMS delay spread CDF")
    ax.grid()
    ax.legend(loc='lower right')
    if save_name:
        plt.savefig(f"../results/plots/RMS_delay_spread_cdf_{save_name}.eps",
                format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return
    
def compare_RMS_angular_spreads_cdf(dfs, names, save_name=None):
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    names = [names] if not isinstance(names, list) else names
    fig, axes = plt.subplots(1, 2, figsize=(20, 6),sharey=True)
    colors = ['b', 'r', 'g', 'm']
    for i, df in enumerate(dfs):
        az_spreads, el_spreads = compute_RMS_angular_spreads(df)
        x_az, y_az = electromagnetism_utils.cdf(az_spreads)
        x_el, y_el = electromagnetism_utils.cdf(el_spreads)
        
        axes[0].plot(x_az, y_az, '-' + colors[i], label=names[i])
        axes[1].plot(x_el, y_el, '-' + colors[i], label=names[i])
    
    axes[0].set(xlabel="RMS azimuth spread (deg)",
                ylabel='Cumulative probability',
                title="RMS azimuth spread cdf")
    axes[1].set(xlabel="RMS elevation spread (deg)",
                title="RMS elevation spread cdf")
    axes[0].grid(),axes[1].grid() 
    axes[0].legend(loc='lower right'),axes[1].legend(loc="lower right")
    fig.subplots_adjust(wspace=0.05)
    if save_name:
        plt.savefig(f"../results/plots/RMS_angular_spread_cdf_{save_name}.eps",
                    format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return


def compare_angular_windows_cdf(dfs,names, save_name=None):
    #plot the percentage of receivers where the angle spread is under ...degrees
    #percentage of receives where >70% power is received within a ...degrees radius.
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    names = [names] if not isinstance(names, list) else names
    fig, axes = plt.subplots(1, 2, figsize=(20, 6),sharey=True)
    colors = ['b', 'r', 'g', 'm']

    for i, df in enumerate(dfs):
        az_win, el_win = compute_angular_windows(df)
        x_az, y_az = electromagnetism_utils.cdf(az_win)
        x_el, y_el = electromagnetism_utils.cdf(el_win)
        axes[0].plot(x_az, y_az, '-' + colors[i], label=names[i])
        axes[1].plot(x_el, y_el, '-' + colors[i], label=names[i])
    
    axes[0].set(xlabel="Azimuth angular window (deg)",
           ylabel='Cumulative probability',
           title="Azimuth angular window CDF")
    axes[1].set(xlabel="Elevation angular window (deg)",
            title="Elevation angular window CDF")
    axes[0].grid(), axes[1].grid()
    axes[0].legend(loc='lower right'), axes[1].legend(loc="lower right")
    fig.subplots_adjust(wspace=0.05)
    if save_name:
        plt.savefig(f"../results/plots/angular_windows_cdf_{save_name}.eps",
                format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return
        

#for comparison of different antenna positions
def compare_power_percent_coming_from_n_rays_multiple_df(dfs,names,save_name=None, nrays=1):
    dfs = [dfs] if not isinstance(dfs, list) else dfs
    names = [names] if not isinstance(names, list) else names
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['b', 'r', 'g', 'm']
    x_values = np.arange(0, 101, 1)
    for i, df in enumerate(dfs):
        percents = compute_percent_coming_from_n_rays(df, nrays)    
        counts = np.sum(percents > x_values[:, np.newaxis], axis=1)
        #percents > x_values[:, np.newaxis] comparison generates a boolean array of shape (len(x_values), len(percents))
        y_values = (counts / len(percents)) * 100
        ax.plot(x_values, y_values,'-o'+colors[i],label=names[i])
        
    ax.grid(), ax.legend(loc='lower left')
    ax.set(xlabel="Percentage of the total power reaching the receiver",
           ylabel='Percentage of receivers',
           title=f'Proportion of RX where {nrays} rays exceed x% of the total received power')
    if save_name:
        plt.savefig(f"../results/plots/power_percent_{nrays}rays_{save_name}.eps",
                format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return   

#for comparison with different number of rays.
def compare_power_percent_coming_from_n_rays(df,nrays,save_name=None):
    nrays = [nrays] if not isinstance(nrays, list) else nrays
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['b', 'r', 'g', 'm']
    x_values = np.arange(0, 101, 1)
    for i,n in enumerate(nrays):
        percents = compute_percent_coming_from_n_rays(df, n)    
        counts = np.sum(percents > x_values[:, np.newaxis], axis=1)
        #percents > x_values[:, np.newaxis] comparison generates a boolean array of shape (len(x_values), len(percents))
        y_values = (counts / len(percents)) * 100
        ax.plot(x_values, y_values,'-'+colors[i],label=f"{n} rays")
    ax.grid(), ax.legend(loc='lower left')
    ax.set(xlabel="Percentage of the total power reaching the receiver",
           ylabel='Percentage of receivers',
           title=f'Proportion of RX where {nrays} rays exceed x% of the total received power')
    if save_name:
        plt.savefig(f"../results/plots/power_percent_rays_comp_{save_name}.eps",
                format='eps', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()
    return   

if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close('all')
    # df_path = "../results/place_saint_jean_160p_em_solved.csv"
    # df = electromagnetism_utils.load_df(df_path)

    #plot_rx_rays_distribution(df, receivers=np.arange(0, 1, 1), save_name="test_rays_distrib")
    #plot_rays_on_sphere(df, receivers=[1,2,3], save_name=None)
    # plot_order_importance(df_path)
    # plot_measures_only()
    # EM_fields_plots(df_path)
    

    
    df1_path = "../results/plots/solved/data/slanted_place_saint_jean_tx_at_centraal_O2_12_160p_em_solved.csv"
    df1 = electromagnetism_utils.load_df(df1_path)
    name1="on_top_of_48m_building"
    
    df2_path = "../results/plots/solved/data/slanted_place_saint_jean_tx_in_middle_O2_12_160p_em_solved.csv"
    df2 = electromagnetism_utils.load_df(df2_path)
    name2="in_the_intersection"
    dfs=[df1,df2]
    names=[name1,name2]
    
    #compare_RMS_delay_spreads_cdf(dfs,names)
    
    #compare_power_percent_coming_from_n_rays(dfs,names,nrays=1)
    
    compare_power_percent_coming_from_n_rays(df2,nrays=[1,2,3,4],save_name="tx_in_middle")
    
    #compare_angular_windows_cdf(dfs,names)
    #compare_RMS_angular_spreads_cdf(dfs,names)

    #compare_RMS_delay_spreads_pdf(dfs,names)

    #plot_rays_on_sphere(df1,receivers=[1,2,3],save_name='test')

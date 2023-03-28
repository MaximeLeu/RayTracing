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
    df=file_utils.load_df(df_path)
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


if __name__=='__main__':
    plt.close('all')
    
    
    
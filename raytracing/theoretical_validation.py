#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:49:38 2022

Comparaison between the two rays model and the simulator,
also includes comparaisons with a simplified two rays model, and path loss

@author: Maxime Leurquin
"""
#packages
import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt

from ray_tracing import RayTracingProblem
import place_utils
import plot_utils

from electromagnetism import my_field_computation,ElectromagneticField,\
     TX_GAIN,RX_GAIN, P_IN,Antenna,vv_normalize, to_db, path_loss
from materials_properties import FREQUENCY,DF_PROPERTIES,LAMBDA,K,Z_0

def simplified_two_rays(L,ztx,zrx):
    """
    Simplified two rays model, only valid when the distance between tx and rx is large
    """
    #http://www.wirelesscommunication.nl/reference/chaptr03/pel/tworay.htm
    ans  = RX_GAIN*TX_GAIN*((LAMBDA/(4*np.pi*L))**2)* 4 *(np.sin(2*np.pi*zrx*ztx/(LAMBDA*L)))**2
    ans =10*np.log10(ans)
    return ans

def two_rays_geometry(L,ztx,zrx):
    """
    L is the distance between the bases of the antennas.
    Ztx is TX antenna height, zrx is RX antenna height
    """
    #geometry
    dlos=np.sqrt((ztx-zrx)**2+L**2)
    dref=np.sqrt((ztx+zrx)**2+L**2) #d1+d2
    d1=ztx*dref/(ztx+zrx) #thales
    d2=dref-d1
    L1=L*ztx/(ztx+zrx) #thales
    L2=L-L1
    theta=np.arctan2(ztx+zrx,L)
    theta_tx=np.arcsin(2*zrx*L/(dlos*dref))#law of sines
    theta_rx=2*theta-theta_tx
    
    assert(0<theta_rx<pi/2), f'antennas are too close theta_RX {theta_rx*180/pi}\u00b0 L={L}m L2={L2}m theta={theta*180/pi}\u00b0'
    assert(0<theta_tx<pi/2), f'antennas are too close theta_TX {theta_tx*180/pi}\u00b0 L={L}m L2={L2}m theta={theta*180/pi}\u00b0'
    return dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx



def two_rays_fields(L,ztx,zrx):
    """
    L is the distance between the bases of the antennas.
    ztx is TX antenna height, zrx is RX antenna height
    """
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx= two_rays_geometry(L,ztx,zrx)
    #csts
    epsilon_eff_2=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=='concrete']["epsilon_eff"].values[0]
    gamma_par,gamma_per=ElectromagneticField.fresnel_coeffs(pi/2-theta, 0, epsilon_eff_2)

    #definitions:
    tx=np.array([0,0,ztx])
    rx=np.array([0,L,zrx])
    g=np.array([0,L1,0])
    #vectors
    dlos_vv=vv_normalize(rx-tx)# tx-->rx
    d1_vv=vv_normalize(g-tx)#tx-->g
    d2_vv=vv_normalize(rx-g)#g-->rx

    #REFERENCE FRAMES:
    #world
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.array([0,0,1])

    #tx antenna
    tx_z=dlos_vv #TODO: modify schematic accordingly
    tx_x=x
    tx_y=vv_normalize(np.cross(tx_x,tx_z))

    #rx antenna
    rx_z=-dlos_vv #TODO: modify schematic accordingly
    rx_x=x
    rx_y=-vv_normalize(np.cross(rx_x,rx_z))
    
    tx_basis=[tx_x,tx_y,tx_z]
    rx_basis=[rx_x,rx_y,rx_z]
    
    def show_antennas_alignement():
        """
        shows the TX and RX antennas reference frames, and the world frame
        some vectors may not appear to be unit vectors, however they are:
        its just matplotlib scaling the axes.
        """
        def plot_antenna_frame(antenna_basis,antenna_position,ax,colors):
            """
            plots the antenna reference frame on ax
            """
            for i in range(3):
                plot_utils.plot_vec(ax,antenna_basis[i],colors[i],origin=antenna_position)
            return ax
        fig=plt.figure()
        fig.set_dpi(300)
        ax=fig.add_subplot(1,1,1,projection='3d')
        colors=['r','g','b']
        ax=plot_utils.plot_world_frame(ax, colors)
        ax=plot_antenna_frame(tx_basis,tx,ax,colors)
        ax=plot_antenna_frame(rx_basis,rx,ax,colors)
        ax.legend()
        plt.show()
        return
    #show_antennas_alignement()
    
    def vv_TX2W(vv):
        tx_mat=np.array([
                    [np.dot(tx_x,x),np.dot(tx_x,y),np.dot(tx_x,z)],
                   [np.dot(tx_y,x),np.dot(tx_y,y),np.dot(tx_y,z)],
                   [np.dot(tx_z,x),np.dot(tx_z,y),np.dot(tx_z,z)],
                   ])
        new_vv=vv@tx_mat
        return new_vv

    def vv_W2RX(vv):
        rx_mat=np.array([
                    [np.dot(rx_x,x),np.dot(rx_x,y),np.dot(rx_x,z)],
                   [np.dot(rx_y,x),np.dot(rx_y,y),np.dot(rx_y,z)],
                   [np.dot(rx_z,x),np.dot(rx_z,y),np.dot(rx_z,z)],
                   ])
        new_vv=vv@rx_mat.T
        # print(f'{vv_W2RX(rx_x)} should have 100')
        # print(f'{vv_W2RX(rx_y)} should have 010')
        # print(f'{vv_W2RX(rx_z)} should have 001')
        # print(f'{vv_W2RX(-d2_vv)} should have {np.array([0,-np.sin(theta_rx),np.cos(theta_rx)])}')
        return new_vv
    
    #E0=-1j*K*Z_0*np.sqrt(2*Z_0*TX_GAIN*P_IN/(4*pi))*1/(4*pi)*np.exp(-1j*K*1)*tx_y
    
    
    E0=-1j*Z_0*np.exp(-1j*K*1)*tx_y
    Elos=E0*np.sqrt(Antenna.radiation_pattern(0))*np.exp(-1j*K*dlos)/dlos
    per_vv=vv_normalize(np.cross(d1_vv,d2_vv))
    par_vv=vv_normalize(np.cross(per_vv,d1_vv))
    
    #E0ref=E0*np.sqrt(Antenna.radiation_pattern(theta_tx))
    E0ref=E0*np.sqrt(Antenna.radiation_pattern(theta_tx)) #TODO why take sqrt of radiation pattern
    Eref=E0ref*np.exp(-1j*K*(d1+d2))/(d1+d2)*(gamma_par*np.dot(tx_y,par_vv)*par_vv \
                                           +gamma_per*np.dot(tx_y,per_vv)*per_vv)

    #put the fields relative to the rx frame 
    #in_tx_frame=np.real(Elos)
    Elos=vv_TX2W(Elos)
    Eref=vv_TX2W(Eref)
    #in_world_frame=np.real(Elos)
    Elos=vv_W2RX(Elos)
    Eref=vv_W2RX(Eref)
    #in_rx_frame=(np.real(Elos))
    #print(f"Elos in rx frame {in_rx_frame} and in tx frame {in_tx_frame} in world_frame {in_world_frame}")
    
    los_power=1/(2*Z_0)*Antenna.compute_Ae(0)*(np.linalg.norm(np.real(Elos)))**2
    ref_power=1/(2*Z_0)*Antenna.compute_Ae(theta_rx)*(np.linalg.norm(np.real(Eref)))**2
    P_rx=los_power+ref_power
        
    db=to_db(P_rx)
    assert(np.linalg.norm(Eref)<np.linalg.norm(Elos))
    return db



def compare_models():
    """
    Comparison between path loss, the two rays model, and the simplified two rays model
    """
    ztx=30
    zrx=2
    dists=np.arange(5*ztx*zrx,500*1e3,100) #distances between the bases of the antennas

    pl=np.zeros(len(dists))
    sol_simplified_two_rays=np.zeros(len(dists))
    sol_two_rays_fields=np.zeros(len(dists))

    for ind,L in enumerate(dists):
        dlos=np.sqrt((ztx-zrx)**2+L**2)
        pl[ind]=path_loss(dlos)
        sol_two_rays_fields[ind]=two_rays_fields(L=L,ztx=ztx,zrx=zrx)
        sol_simplified_two_rays[ind]=simplified_two_rays(L=L,ztx=ztx,zrx=zrx)


    fig=plt.figure()
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)
    ax.set_xscale('log')
    ax.plot(dists,pl,'b',label="path loss")
    ax.plot(dists,sol_simplified_two_rays,'y',label="simplified two rays")
    ax.plot(dists,sol_two_rays_fields,'r',label='two rays')

    ax.grid()
    ax.set_title('comparison between two rays and path loss')
    ax.set_xlabel('distance between tx and rx bases (m)')
    ax.set_ylabel(r'Received power ($p_{rx}/p_{tx}$)[dB]')
    ax.legend()
    plt.show()
    return ax

def compare_two_rays_and_simu(npoints):
    place,tx,geometry=place_utils.create_two_rays_place(npoints,plot=True)
    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=2,receivers_indexs=None)
    problem.plot_all_rays()
    results_path=f'../results/{geometry}_launch.csv'

    #simu solve
    df=my_field_computation(problem,results_path)
    nreceivers=len(df['rx_id'].unique())
    sol_two_rays_fields=np.zeros(nreceivers)
    simu=np.zeros(nreceivers)
    dists=np.zeros(nreceivers)

    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        simu[receiver]=to_db(np.sum(rx_df['path_power']))
        rx_base=np.array([rx_coord[0],rx_coord[1],0])
        tx_base=np.array([tx[0][0],tx[0][1],0])
        dists[receiver]=np.linalg.norm(tx_base-rx_base)
        sol_two_rays_fields[receiver]=two_rays_fields(L=dists[receiver],ztx=tx[0][2],zrx=rx_coord[2])

    fig=plt.figure()
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)

    ax.plot(dists,sol_two_rays_fields,'-or',label='two rays')
    ax.plot(dists,simu,'-og',label="simu")

    ax.grid()
    ax.set_title('comparison between two rays and simu')
    ax.set_xlabel('distance between tx and rx (m)')
    ax.set_ylabel(r'Received power ($p_{rx}/p_{tx}$)[dB]')
    ax.legend()
    plt.show()
    plt.savefig("../plots/comparison_two_rays_simu.eps", format='eps', dpi=1000)
    print(f'TXgain*RXgain={np.sqrt(TX_GAIN*RX_GAIN):.2f}')
    return


if __name__=='__main__':

    plt.close('all')
    compare_two_rays_and_simu(npoints=100)
    compare_models()

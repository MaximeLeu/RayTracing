#!/usr/bin/env python3
#pylint: disable=invalid-name,line-too-long
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:49:38 2022

Testing two rays model vs my simulator

@author: max
"""
#packages
import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt


from ray_tracing import RayTracingProblem
import place_utils

from electromagnetism import my_field_computation,ElectromagneticField,\
    RADIATION_POWER, TX_GAIN,RX_GAIN,Antenna,vv_normalize, to_db
from materials_properties import FREQUENCY,DF_PROPERTIES,LAMBDA,K,Z_0


Pin=1


def compute_path_loss(d):
    pr_pt=(RX_GAIN*TX_GAIN*(LAMBDA/(4*pi*d))**2)
    pl=10*np.log10(pr_pt)
    return pl

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
    
    assert(0<theta_rx<pi/2), f'antennas are too close theta_RX {theta_rx*180/pi} L {L} L2 {L2} theta {theta}'
    assert(0<theta_tx<pi/2), f'antennas are too close theta_TX {theta_tx*180/pi} L {L} L2 {L2} theta {theta}'
    return dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx


def radiation_pattern(theta):
    #its the same for RX and TX
    pattern=np.cos(theta)
    return pattern

def compute_Ae(theta_rx):
    Ae=(LAMBDA**2/(4*pi))*RX_GAIN*radiation_pattern(theta_rx)
    return Ae

def two_rays_fields_1(L,ztx,zrx):
    """
    L is the distance between the bases of the antennas.
    ztx is TX antenna height, zrx is RX antenna height
    """
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx= two_rays_geometry(L,ztx,zrx)
    #csts
    epsilon_eff_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=='air']["epsilon_eff"].values[0]
    epsilon_eff_2=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=='concrete']["epsilon_eff"].values[0]
    epsilon_ratio=epsilon_eff_2/epsilon_eff_1

    #fresnel
    sqrt=np.sqrt(epsilon_ratio-np.sin(theta)**2)
    gamma_par=np.linalg.norm((epsilon_ratio*np.cos(theta)-sqrt)/((epsilon_ratio*np.cos(theta)+sqrt)))
    gamma_per=np.linalg.norm((np.cos(theta)-sqrt)/(np.cos(theta)+sqrt))


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

    #antenna
    new_z=dlos_vv #TODO: modify schematic accordingly
    new_x=x#vv_normalize(np.cross(np.array([0,0,1]),new_z))
    new_y=vv_normalize(np.cross(new_x,new_z))


    mat=np.array([
                [np.dot(new_x,x),np.dot(new_x,y),np.dot(new_x,z)],
               [np.dot(new_y,x),np.dot(new_y,y),np.dot(new_y,z)],
               [np.dot(new_z,x),np.dot(new_z,y),np.dot(new_z,z)],
               ])

    def pp_A2W(point,antenna_position):
        new_point=point@mat
        return new_point+antenna_position

    def pp_W2A(point,antenna_position):
        tr_point=point-antenna_position
        new_point=tr_point@mat.T
        return new_point

    def vv_W2A(vv):
        new_vv=vv@mat.T
        return new_vv

    def vv_A2W(vv):
        new_vv=vv@mat
        return new_vv


    # print()
    # print(f'{pp_W2A(tx,tx)} should have 000')
    # print(f'{pp_W2A(rx,tx)} should have 00{dlos}')

    # print(f'{vv_W2A(new_x)} should have 100')
    # print(f'{vv_W2A(new_y)} should have 010')
    # print(f'{vv_W2A(new_z)} should have 001')
    # print(f'{vv_W2A(d1_vv)} should have {np.array([0,np.sin(theta_tx),np.cos(theta_tx)])}')

#TODO: problem: E0 is not a vector but should be.
    E0=-1j*K*Z_0*np.sqrt(2*Z_0*TX_GAIN*Pin/(4*pi))*1/(4*pi)*np.exp(-1j*K*1)*new_y

    Elos=E0*np.exp(-1j*K*dlos)/dlos*new_y
    #Elos=vv_A2W(Elos, tx)

    per_vv=vv_normalize(np.cross(d1_vv,d2_vv))
    par_vv=vv_normalize(np.cross(per_vv,d1_vv))
    E0ref=E0*np.sqrt(radiation_pattern(theta_tx))

    Eref=E0ref*np.exp(-1j*K*(d1+d2))/(d1*d2)*(gamma_par*np.dot(new_y,par_vv)*par_vv \
                                             +gamma_per*np.dot(new_y,per_vv)*per_vv)

    #Eref=vv_A2W(Eref, tx)

    P_rx=1/(2*Z_0)*(compute_Ae(0)*(np.linalg.norm(np.real(Elos)))**2+compute_Ae(theta_rx)*(np.linalg.norm(np.real(Eref)))**2)
    db=10*np.log10(P_rx/FREQUENCY)

    assert(np.linalg.norm(Eref)<np.linalg.norm(Elos))
    return db


def matlab(L,ztx,zrx):
    #http://www.wirelesscommunication.nl/reference/chaptr03/pel/tworay.htm
    ans  = RX_GAIN*TX_GAIN*((LAMBDA/(4*np.pi*L))**2)* 4 *(np.sin(2*np.pi*zrx*ztx/(LAMBDA*L)))**2
    ans =10*np.log10(ans)
    return ans


def compare_models():
    ztx=30
    zrx=5
    dists=np.arange(30,500*1e3,100) #distances between the bases of the antennas

    pl=np.zeros(len(dists))
    sol_matlab=np.zeros(len(dists))
    sol_two_rays_fields_1=np.zeros(len(dists))

    for ind,L in enumerate(dists):
        dlos=np.sqrt((ztx-zrx)**2+L**2)
        pl[ind]=compute_path_loss(dlos)
        sol_two_rays_fields_1[ind]=two_rays_fields_1(L=L,ztx=ztx,zrx=zrx)
        sol_matlab[ind]=matlab(L=L,ztx=ztx,zrx=zrx)


    fig=plt.figure()
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)

    dists=10*np.log10(dists)
    ax.plot(dists,pl,'b',label="path loss")
    ax.plot(dists,sol_matlab,'y',label="matlab")

    ax.plot(dists,sol_two_rays_fields_1,'r',label='two rays 1')

    ax.grid()
    ax.set_title('comparison between two rays and path loss')
    ax.set_xlabel('distance between tx and rx bases 10log10 (m)')
    ax.set_ylabel('Received power (pr/pt)[dB]')
    ax.legend()
    plt.show()
    return ax



if __name__ == '__main__':


    plt.close('all')
    place,tx,geometry=place_utils.create_two_rays_place(npoints=50,plot=True)
    rx=place.set_of_points
    #print(f'rx {rx} first {rx[1]}')
    compare_models()


    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=2,receivers_indexs=None)
    problem.plot_all_rays()
    results_path=f'../results/{geometry}_launch.csv'

    #simu solve
    df=my_field_computation(problem,results_path)

    nreceivers=len(df['rx_id'].unique())
    sol_two_rays_fields_1=np.zeros(nreceivers)
    simu=np.zeros(nreceivers)
    dists=np.zeros(nreceivers)
    for receiver in range(nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        simu[receiver]=to_db(np.sum(rx_df['path_power'])/FREQUENCY)
        dists[receiver]=np.linalg.norm(tx-rx_coord)

        print(f'RX {receiver}')
        sol_two_rays_fields_1[receiver]=two_rays_fields_1(L=dists[receiver],ztx=tx[0][2],zrx=rx_coord[2])


    fig=plt.figure()
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)

    ax.plot(dists,sol_two_rays_fields_1,'r',label='two rays 1')
    ax.plot(dists,simu,'g',label="simu")

    ax.grid()
    ax.set_title('comparison between two rays and simu')
    ax.set_xlabel('distance between tx and rx (m)')
    ax.set_ylabel('Received power (pr/pt)[dB]')
    ax.legend()
    plt.show()
    
    print(f'TX gain*rxGain {np.sqrt(TX_GAIN*RX_GAIN)}')

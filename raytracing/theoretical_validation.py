#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:49:38 2022

Testing two rays model vs my simulator

@author: max
"""

from ray_tracing import RayTracingProblem
import place_utils
from electromagnetism import my_field_computation,ElectromagneticField
import plot_utils
#packages
import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt

#constants:
from electromagnetism import RADIATION_POWER, TX_GAIN,RX_GAIN,Antenna,vv_normalize, to_db
from materials_properties import FREQUENCY,DF_PROPERTIES


Z_0=120*pi
lam=c/FREQUENCY
k=2*pi/lam
Pin=1


def compute_path_loss(d):
    pr_pt=(RX_GAIN*TX_GAIN*(lam/(4*pi*d))**2)
    pl=10*np.log10(pr_pt)
    return pl

def two_rays_geometry(L,ztx,zrx):
    #geometry
    dlos=np.sqrt((ztx-zrx)**2+L**2)
    dref=np.sqrt((ztx+zrx)**2+L**2) #d1+d2
    d1=ztx*dref/(ztx+zrx) #thales
    d2=dref-d1
    L1=L*ztx/(ztx+zrx) #thales
    L2=L-L1
    theta=np.arctan2(ztx+zrx,L)
    theta_tx=np.arcsin(d2/dlos*np.sin(2*theta)) #law of sines
    theta_rx=2*theta-theta_tx
    print(f"two rays angle theta {theta_rx*180/pi}")
    assert(theta_rx>0 and theta_rx<pi/2), f'antennas are too close theta_RX {theta_rx*180/pi} L {L} L2 {L2} theta {theta}' 
    assert(theta_tx>0 and theta_tx<pi/2), f'antennas are too close theta_TX {theta_tx*180/pi} L {L} L2 {L2} theta {theta}'
    
    return dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx
    
    
def compute_E0(theta_tx): 
    if theta_tx>pi/2:
        print("PORBLEM 2")
    E0=np.exp(-1j*k*1)*np.sqrt(2*Z_0*TX_GAIN*np.sin(pi/2+theta_tx)*Pin/4*pi) 
    return E0

def compute_Ae(theta_rx):
    Ae=(lam**2/(4*pi))*RX_GAIN*np.sin(theta_rx+pi/2)
    return Ae    
    
def two_rays_fields_1(L,ztx,zrx):
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx= two_rays_geometry(L,ztx,zrx)
    #csts 
    epsilon_eff_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=='air']["epsilon_eff"].values[0]
    epsilon_eff_2=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=='concrete']["epsilon_eff"].values[0]
    epsilon_ratio=epsilon_eff_2/epsilon_eff_1
    
    
    #definitions:
    tx=np.array([0,0,ztx])
    rx=np.array([0,L,zrx])
    g=np.array([0,L1,0])
    #vectors
    dlos_vv=vv_normalize(rx-tx)
    d1_vv=vv_normalize(g-tx)
    d2_vv=vv_normalize(rx-g)
    
    #antenna
    ez_vv=dlos_vv
    ex_vv=np.cross(np.array([0,0,1]),ez_vv)
    ey_vv=np.cross(ex_vv,ez_vv)
    
    #fresnel
    sqrt=np.sqrt(epsilon_ratio-np.sin(theta)**2)
    gamma_par=np.linalg.norm((epsilon_ratio*np.cos(theta)-sqrt)/((epsilon_ratio*np.cos(theta)+sqrt)))
    gamma_per=np.linalg.norm((np.cos(theta)-sqrt)/(np.cos(theta)+sqrt))   
    

    E0=-1j*k*Z_0*np.exp(-1j*k*1)*np.sqrt(2*Z_0*TX_GAIN*Pin/(4*pi)) #radiation pattern added after
    
    E0los=E0*1*ey_vv
    Elos=E0los*np.exp(-1j*k*dlos)/dlos
    
    horrible_par_vv=np.cross(np.cross(d1_vv,d2_vv),d1_vv)
    E0ref=E0*np.sqrt(np.sin(pi/2+theta_tx))
    Eref=E0ref*np.exp(-1j*k*(d1+d2))/(d1*d2)*(gamma_par*np.dot(ey_vv,horrible_par_vv)*horrible_par_vv \
                                             +gamma_per*np.dot(ey_vv,np.cross(d1_vv,d2_vv))*(np.cross(d1_vv,d2_vv)))
        
    P_rx=1/(2*Z_0)*(compute_Ae(0)*(np.linalg.norm(np.real(Elos)))**2+compute_Ae(theta_rx)*(np.linalg.norm(np.real(Eref)))**2)
    db=10*np.log10(P_rx/FREQUENCY)
    
    assert(np.linalg.norm(Eref)<np.linalg.norm(Elos)) 
    return db
    

def matlab(L,ztx,zrx):
    #http://www.wirelesscommunication.nl/reference/chaptr03/pel/tworay.htm
    matlab  = RX_GAIN*TX_GAIN*((lam/(4*np.pi*L))**2)* 4 *(np.sin(2*np.pi*zrx*ztx/(lam*L)))**2
    matlab =10*np.log10(matlab)
    return matlab


def compare_models():
    ztx=30
    zrx=5
    dists=np.arange(30,500*1e3,100)
    
    pl=np.zeros(len(dists))
    sol_matlab=np.zeros(len(dists))
    sol_two_rays_fields_1=np.zeros(len(dists))
    
    for d in range(0,len(dists)):
        pl[d]=compute_path_loss(dists[d])
        sol_two_rays_fields_1[d]=two_rays_fields_1(L=dists[d],ztx=ztx,zrx=zrx)
        sol_matlab[d]=matlab(L=dists[d],ztx=ztx,zrx=zrx)
    
    fig=plt.figure()        
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)
    
    dists=10*np.log10(dists)
    ax.plot(dists,pl,'b',label="path loss")
    ax.plot(dists,sol_matlab,'y',label="matlab")
    
    ax.plot(dists,sol_two_rays_fields_1,'r',label='two rays 1')
    
    ax.grid()
    ax.set_title('comparison between two rays and path loss')
    ax.set_xlabel('distance between tx and rx 10log10 (m)')
    ax.set_ylabel('Received power (pr/pt)[dB]')
    ax.legend()
    plt.show() 
    return ax

    

if __name__ == '__main__':
    
    
    plt.close('all')
    place,tx,geometry=place_utils.create_two_rays_place(npoints=30)
    rx=place.set_of_points
    print(f'rx {rx} first {rx[1]}')
    #compare_models()
     

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
    
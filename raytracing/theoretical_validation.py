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
from electromagnetism import RADIATION_POWER, TX_GAIN,RX_GAIN,Antenna,vv_normalize
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
    
def two_rays_fields_2(L,ztx,zrx):
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx= two_rays_geometry(L,ztx,zrx)
    
    Elos=compute_E0(0)*np.exp(-1j*k*dlos)/dlos
    gamma_par=-1
    
    Eref=gamma_par*compute_E0(theta_tx)*np.exp(-1j*k*dref)/dref
    Prx=compute_Ae(0)*np.linalg.norm(np.real(Elos))**2+compute_Ae(theta_rx)*np.linalg.norm(np.real(Eref))**2
    db=10*np.log10(Prx)
    return db
    
def two_rays_fields_3(L,ztx,zrx):
    #https://arxiv.org/pdf/2001.06459.pdf
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx= two_rays_geometry(L,ztx,zrx)
    gamma_par=-1
    if theta_tx>pi/2:
        print("shitt 2")
    test=Pin*((lam/(4*pi))**2)*np.linalg.norm( \
        compute_E0(0)/dlos +\
        compute_E0(theta_tx)*gamma_par*np.real(np.exp(-1j*k*(dref-dlos)/dref)))**2  #np.sqrt(TX_GAIN*np.sin(pi/2+theta_tx)*RX_GAIN*np.sin(pi/2+theta_rx))
    

    db=10*np.log10(test)
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
    sol_two_rays_fields_2=np.zeros(len(dists))
    sol_two_rays_fields_3=np.zeros(len(dists))
    
    for d in range(0,len(dists)):
        pl[d]=compute_path_loss(dists[d])
        sol_two_rays_fields_1[d]=two_rays_fields_1(L=dists[d],ztx=ztx,zrx=zrx)
        sol_two_rays_fields_2[d]=two_rays_fields_2(L=dists[d],ztx=ztx,zrx=zrx)
        sol_two_rays_fields_3[d]=two_rays_fields_3(L=dists[d],ztx=ztx,zrx=zrx)
        sol_matlab[d]=matlab(L=dists[d],ztx=ztx,zrx=zrx)
    
    fig=plt.figure()        
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)
    
    dists=10*np.log10(dists)
    ax.plot(dists,pl,'b',label="path loss")
    ax.plot(dists,sol_matlab,'y',label="matlab")
    
    ax.plot(dists,sol_two_rays_fields_1,'r',label='two rays 1')
    ax.plot(dists,sol_two_rays_fields_2,'g',label="two rays 2")
    ax.plot(dists,sol_two_rays_fields_3,'orange',label="two rays 3")
    
    
    ax.grid()
    ax.set_title('comparison between two rays and path loss')
    ax.set_xlabel('distance between tx and rx (m)')
    ax.set_ylabel('Received power (pr/pt)[dB]')
    ax.legend()
    plt.show() 
    return ax

    

if __name__ == '__main__':
    
    
    plt.close('all')
    place,tx,geometry=place_utils.create_two_rays_place()
    rx=place.set_of_points
    print(f'rx {rx} first {rx[1]}')
    compare_models()
     

    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=2,receivers_indexs=None)
    problem.plot_all_rays()
    results_path=f'../results/{geometry}_launch.csv'
    
    #simu solve
    df=my_field_computation(problem,results_path)
    simu_los=df.loc[df['path_type'] == "LOS"]['field_strength'].values[0]   
    simu_ref=df.loc[df['path_type'] == "R"]['field_strength'].values[0] 
    
    sol_two_rays_fields_1=np.zeros(len(simu_los))
    simu=np.zeros(len(simu_los))
    dists=np.zeros(len(simu_los))
    for i in range(len(simu_los)):
        rx_power=1/(2*Z_0)*(compute_Ae(0)*(np.linalg.norm(np.real(simu_los[i])))**2\
                            +compute_Ae(pi/8)*(np.linalg.norm(np.real(simu_ref[i])))**2)
        simu[i]=10*np.log10(rx_power)
        rx=place.set_of_points[i]        
        dists[i]=np.linalg.norm(tx-rx)
        sol_two_rays_fields_1[i]=two_rays_fields_1(L=dists[i],ztx=tx[0][2],zrx=rx[2])
    
    fig=plt.figure()        
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1)
    
    ax.plot(dists,sol_two_rays_fields_1,'r',label='two rays 1')
    ax.plot(dists,simu,'g',label="simu")
    
    ax.grid()
    ax.set_title('comparison between two rays and path loss')
    ax.set_xlabel('distance between tx and rx (m)')
    ax.set_ylabel('Received power (pr/pt)[dB]')
    ax.legend()
    plt.show() 
    
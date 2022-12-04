#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:49:38 2022

Testing two rays model vs my simulator

@author: max
"""

from ray_tracing import RayTracingProblem
import place_utils
from electromagnetism import my_field_computation,EM_fields_plots,EM_fields_data,ElectromagneticField

import geometry as geom
#packages
import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
import csv

#constants:
from electromagnetism import RADIATION_POWER, TX_GAIN,RX_GAIN
from materials_properties import FREQUENCY,DF_PROPERTIES
Z_0=120*pi
lam=c/FREQUENCY
k=2*pi/lam


def vv_normalise(vv):
    return vv/np.linalg.norm(vv)


def theorical_solve(los,reflection):
    ref_point=reflection[1]
    tx=los[0]
    rx=los[1]
    
    
    
    #computing various distances
    d_los=np.linalg.norm(rx-tx) #dist tx-rx
    L=np.linalg.norm(tx[0:2]-rx[0:2]) #dist base of tx to base of rx
    L1=tx[2]/rx[2]*L/(1+tx[2]/rx[2]) #dist base of tx to ground reflection
    L2=L-L1#dist base of rx to ground reflection
    d1=np.sqrt(L1**2+tx[2]**2) #dist tx-ground reflection
    d2=np.sqrt(L2**2+rx[2]**2) #dist rx-ground reflection
    theta=np.arcsin(tx[2]/d1)#incident angle to the ground
    
    E0_los=ElectromagneticField.compute_E0(los)
    E0_ref=ElectromagneticField.compute_E0(reflection)
    
    #separating the // and perp components of the E-field
    
    
    tx_to_ref_vv=vv_normalise(reflection[1]-tx) 
    
    
    ref_to_rx_vv=vv_normalise(rx-reflection[1])
    vv_per=vv_normalise(np.cross(tx_to_ref_vv,ref_to_rx_vv))
    vv_par=vv_normalise(np.cross(vv_per,tx_to_ref_vv))
    
    print(f"dot {np.dot(vv_par,E0_ref.E)}")
    E0_par= np.dot(vv_par,E0_ref.E)*vv_par
    E0_per= np.dot(vv_per,E0_ref.E)*vv_per
    E0_other=np.dot(tx_to_ref_vv,E0_ref.E)*tx_to_ref_vv
    assert(np.around(np.linalg.norm(E0_other),decimals=2)==0),f"ERROR E field component in the direction of propagation is non zero: E_s={E0_other}, but should be [0,0,0]"
    assert (E0_par+E0_per==E0_ref.E).all(), f"E0_per {E0_per+E0_par} E0_ref{E0_ref.E}"
    
    #fresnel
    epsilon_eff=DF_PROPERTIES.loc[DF_PROPERTIES['material'] == "concrete"]['epsilon_eff'].values[0]   
    roughness=DF_PROPERTIES.loc[DF_PROPERTIES['material'] == "concrete"]['roughness'].values[0]
    r_par,r_per=ElectromagneticField.fresnel_coeffs(E0_ref,theta,roughness,epsilon_eff)
    
    
    E_los=E0_los.E/d_los*np.exp(-1j*k*d_los)
    E_ref=  (r_par*E0_ref.E+r_per*E0_ref.E+tx_to_ref_vv*E0_ref)    *1/(d1+d2)*np.exp(-1j*k*(d1+d2))
    
    #sanity checks
    assert np.around(np.linalg.norm(ref_point[0:2]-tx[0:2]),decimals=1)==np.around(L1,decimals=1), f"L1 {L1} real L1 {np.linalg.norm(ref_point[0:2]-tx[0:2])}"
    assert np.around(np.linalg.norm(ref_point[0:2]-rx[0:2]),decimals=1)==np.around(L2,decimals=1)
    assert np.around(geom.path_length(reflection),decimals=1)==np.around(d1+d2,decimals=1)
    assert np.arcsin(tx[2]/d1)==np.arcsin(rx[2]/d2) #incident angle=reflection angle
    
    
    return E_los,E_ref


    


    

if __name__ == '__main__':
    
    plt.close('all')
    place,tx,geometry=place_utils.create_two_rays_place()
    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=2,receivers_indexs=None)
    problem.plot_all_rays()
    
    results_path=f'../results/{geometry}_launch.csv'
    df=my_field_computation(problem,results_path)
    
    
    simu_los=df.loc[df['path_type'] == "LOS"]['field_strength'].values[0]   
    simu_ref=df.loc[df['path_type'] == "R"]['field_strength'].values[0] 
    
    #theorical solve:
    los=problem.los[0][0]
    reflection=problem.reflections[0][1][0][0]
    E_los,E_ref=theorical_solve(los,reflection)

    print("----------------COMPARISON SIMU AND TWO RAYS FIELDS--------------")
    print(f"simu_los {simu_los} two rays los {E_los}")
    print(f"simu_ref {simu_ref} two rays ref {E_ref}")
    
    print("----------------COMPARISON SIMU AND TWO RAYS NORMS--------------")
    print(f"simu_los {np.linalg.norm(simu_los)} two rays los {np.linalg.norm(E_los)}")

    print(f"simu_ref {np.linalg.norm(simu_ref)} two rays ref {np.linalg.norm(E_ref)}")
    
    
    
    
    # EM_fields_plots(results_path,order=1,name=geometry)    
    # EM_fields_data(results_path)
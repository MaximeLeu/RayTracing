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




def theorical_solve(reflection):
    G=reflection[1]
    tx=reflection[0]
    tx_base=np.array([tx[0],tx[1],0])
    rx=reflection[-1]
    
    d1=np.linalg.norm(G-tx)
    d2=np.linalg.norm(rx-G)
    d_los=np.linalg.norm(rx-tx)
    
    L1=np.linalg.norm(tx_base-G)
    theta=np.arccos(L1/d1)
    
    tx_antenna=Antenna()
    tx_antenna.position=tx
    tx_antenna.align_antenna_to_point(rx)
    
    #small tests
    test=tx_antenna.change_reference_frame(tx)
    print(f"tx {tx} in the new frame {test}")
    new_point=tx_antenna.change_reference_frame(rx)
    print(f"rx {rx} in the new frame {new_point} should have {[0,0,np.linalg.norm(new_point)]}")
     
    
    
    fig=plt.figure()        
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1,projection='3d')
    colors=['r','g','b']
    plot_utils.plot_world_frame(ax, colors)
    plot_utils.plot_path(ax, reflection)
    plot_utils.plot_path(ax, [tx,rx])
    tx_antenna.plot_antenna_frame(ax, colors)
    plt.show()
    
    #init field for los
    d0=1
    F=np.sin(theta)*tx_antenna.basis[1]
    #TODO put r in the basis of the antenna
    
    
    E0=-1j*k*Z_0*np.exp(-1j*k*d0)/(4*pi*d0)*np.cross((np.cross(tx_antenna.basis[2],F)),tx_antenna.basis[2])
    E0=np.real(E0)
    
    #LOS-------------------------------
    Elos=E0*np.exp(-1j*k*d_los)/d_los
    Elos=np.real(Elos)
    
    #REF-----------------------------
    #fresnel
    E=ElectromagneticField()
    epsilon_eff=DF_PROPERTIES.loc[DF_PROPERTIES['material'] == "concrete"]['epsilon_eff'].values[0]   
    roughness=DF_PROPERTIES.loc[DF_PROPERTIES['material'] == "concrete"]['roughness'].values[0]
    r_par,r_per=ElectromagneticField.fresnel_coeffs(E,theta,roughness,epsilon_eff)
    
    #separating the // and perp components
    Ei=E0*np.exp(-1j*k*d1)/d1
    Ei=np.real(Ei)
    Ei_per=np.dot(Ei,tx_antenna.basis[0])*tx_antenna.basis[0]
    Ei_par=np.dot(Ei,tx_antenna.basis[1])*tx_antenna.basis[1]
    
    
    Eref=np.linalg.norm(Ei_per)*r_per*tx_antenna.basis[0]+np.linalg.norm(Ei_par)*r_par*tx_antenna.basis[1]
    Eref=Eref*np.exp(-1j*k*d2)/d2
    Eref=np.real(Eref)
    return Elos,Eref

    

if __name__ == '__main__':
    
    plt.close('all')
    place,tx,geometry=place_utils.create_two_rays_place()
    problem = RayTracingProblem(tx, place)
    problem.solve(max_order=2,receivers_indexs=None)
    problem.plot_all_rays()
    results_path=f'../results/{geometry}_launch.csv'
    
    #simu solve
    df=my_field_computation(problem,results_path)
    simu_los=df.loc[df['path_type'] == "LOS"]['field_strength'].values[0]   
    simu_ref=df.loc[df['path_type'] == "R"]['field_strength'].values[0] 
    
    simu_ref=np.real(simu_ref)
    simu_los=np.real(simu_los)
    
    #theorical solve:
    reflection=problem.reflections[0][1][0][0]
    E_los,E_ref=theorical_solve(reflection)

    print("----------------COMPARISON SIMU AND TWO RAYS FIELDS--------------")
    print(f"simu_los {simu_los} two rays los {E_los}")
    print(f"simu_ref {simu_ref} two rays ref {E_ref}")
    
    print("----------------COMPARISON SIMU AND TWO RAYS NORMS--------------")
    print(f"simu_los {np.linalg.norm(simu_los)} two rays los {np.linalg.norm(E_los)}")

    print(f"simu_ref {np.linalg.norm(simu_ref)} two rays ref {np.linalg.norm(E_ref)}")   
    
    # EM_fields_plots(results_path,order=1,name=geometry)    
    # EM_fields_data(results_path)
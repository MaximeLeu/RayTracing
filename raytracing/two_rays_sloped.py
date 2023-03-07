#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:42:15 2023

@author: max
"""
import matplotlib.pyplot as plt
import numpy as np

L=10
ztx=3
zrx=2
slope=np.pi/50 #slope of the incline


def draw_sloped_two_rays(L,ztx,zrx,slope):
    h=L*np.sin(slope)
    d=np.sqrt(L**2-h**2)

    #ground
    x_ground=[0,d]
    y_ground=[0,-h]

    #antennas
    x_tx=[0,0]
    y_tx=[0,ztx]
    x_rx=[d,d]
    y_rx=[-h,-h+zrx]

    #los
    x_los=[0,d]
    y_los=[ztx,-h+zrx]
    
    dlos,dref,d1,d2,L1,L2,theta,theta_tx,theta_rx=two_rays_geometry_sloped(L,ztx,zrx,slope)
    
    #TX to G
    d1_x=[0,L1*np.cos(slope)]
    d1_y=[ztx,-L1*np.sin(slope)]

    #G to RX
    d2_x=[L1*np.cos(slope),d]
    d2_y=[-L1*np.sin(slope),-h+zrx]
    
    plt.close("all")
    plt.plot(x_ground,y_ground)
    plt.plot(x_tx,y_tx)
    plt.plot(x_rx,y_rx)
    plt.plot(x_los,y_los)
    plt.plot(d1_x,d1_y)
    plt.plot(d2_x,d2_y)
    plt.grid()
    plt.title('Sloped two rays')
    plt.show()

    print(f"Given L={L:.2f}m, alpha={slope*180/np.pi:.2f}\u00b0, ZTX={ztx:.2f}m and ZRX={zrx:.2f}m")
    print(f"d={d:.2f}m, h={h:.2f}m, d1={d1:.2f}m, d2={d2:.2f}m, L1={L1:.2f}m, L2={L2:.2f}m")
    print(f"theta={theta*180/np.pi:.2f}\u00b0, theta_rx={theta_rx*180/np.pi:.2f}\u00b0, theta_tx={theta_tx*180/np.pi:.2f}\u00b0")
    return


draw_sloped_two_rays(L, ztx, zrx, slope)
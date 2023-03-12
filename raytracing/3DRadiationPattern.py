#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:59:36 2023

@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi,c

from Ra.test import import_test

FREQUENCY=12.5*1e9 #in Hz
LAMBDA=c/FREQUENCY

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

#Define the pattern function f here
def f(theta,phi):
    f=abs(np.cos(theta)**4)
    #f=abs(np.sin(theta)*(np.sin(phi))**2)
    return f


def plot_radiation_pattern():
    # Angle Increment
    theta = np.linspace(0, pi, 100)
    phi = np.linspace(0, 2*pi, 100)

    #Compute pattern and normalize
    THETA,PHI=np.meshgrid(theta,phi)
    r=f(THETA,PHI)
    ratio=np.max(np.max(r))
    X,Y,Z=spherical_to_cartesian(r/ratio,THETA,PHI)

    #Plot 3D pattern
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    
    h=ax.plot_surface(X,Y,Z, cmap='jet', edgecolors='black', linewidth=1.1)
    #fig.colorbar(h)
     
    #Plot X,Y,Z axises
    corr = 1.1 * np.max([X, Y, Z])
    ax.plot([0, corr], [0, 0], [0, 0], 'k', linewidth=1.5)
    ax.plot([0, 0], [0, corr], [0, 0], 'k', linewidth=1.5)
    ax.plot([0, 0], [0, 0], [0, corr], 'k', linewidth=1.5)
    ax.text(corr*1.05, 0, 0, 'x', fontsize=12, horizontalalignment='center')
    ax.text(0, corr*1.05, 0, 'y', fontsize=12, horizontalalignment='center')
    ax.text(0, 0, corr*1.05, 'z', fontsize=12, horizontalalignment='center')
    
    # Set Display
    #ax.set_axis_off()
    ax.set_box_aspect([1,1,1])
    ax.view_init(20, -135)
    plt.show()
    return

if __name__=='__main__':
    plt.close("all")
    print(f"LAMBDA {LAMBDA:.5f}m")
    plot_radiation_pattern()
    
    import_test()
    

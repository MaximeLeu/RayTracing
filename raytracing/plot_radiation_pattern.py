#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:17:37 2023

@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy as sc
from matplotlib.widgets import Slider

# Angle Increment
theta = np.linspace(0, pi, 100)
phi = np.linspace(0, 2*pi, 100)


#Define the pattern function f here
def f(theta,phi):
    alpha=5
    f = abs(np.cos(theta/2)**(2*alpha))
    f=abs(np.sin(theta)**2) #DIPOLE ANTENNA
    return f


def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


#3D radiation pattern plot.
def plot_radiation_pattern():
    #Compute pattern and normalize
    THETA,PHI=np.meshgrid(theta,phi)
    r=f(THETA,PHI)
    ratio=np.max(np.max(r))
    X,Y,Z=spherical_to_cartesian(r/ratio,THETA,PHI)

    #Plot 3D pattern
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    h=ax.plot_surface(X,Y,Z, cmap='jet', edgecolors='black', linewidth=1.1)
    plt.title("3D Radiation pattern",fontsize=20)
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


def plot_2D_radiation_pattern():
    
    r=f(theta,phi)
    fig = plt.figure(figsize=(15,10))
    fig.suptitle('Radiation patterns ', fontsize=20)
    
    #The elevation plane pattern is formed by slicing the 3D pattern
    #through an orthogonal plane (either the x-z plane or the y-zplane). 
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.plot(theta,r,linewidth=3)
    ax1.plot(-theta, r,linewidth=3) # mirror the plot to show the other half
    
    #styling
    #ax1.set_thetagrids([0,30, 60,90,120,150,180,210,240,270,300,330])
    #ax1.set_rticks([-3, -10,-20],labels= ["-3dB", "-10dB","-20dB"])
    ax1.set_rlabel_position(135)
    ax1.grid(True,which="minor",linestyle= ":")
    ax1.grid(True,which="major",linewidth= 1.2)
    ax1.minorticks_on()
    plt.title("Elevation plane pattern",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    #The azimuth plane pattern is formed by slicing through the 3D pattern in the horizontal plane, the x-y plane 
    ax2 = fig.add_subplot(122, polar=True,projection='polar')
    ax2.plot(phi,r,linewidth=3)
    ax2.plot(-phi,r,linewidth=3)
    plt.title("Azimuth plane pattern",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.show()
    #plt.savefig(f"../plots/radiation_pattern_alpha{ALPHA}.eps", format='eps', dpi=1000)


if __name__=='__main__':
    plt.close("all")
    plot_radiation_pattern()
    plot_2D_radiation_pattern()
    
    
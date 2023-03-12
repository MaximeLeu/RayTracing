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


def interactive_pattern():
    ALPHA=5
    def rad_pattern(alpha,param):
        F = np.cos(theta/2+param)**(2*alpha)
        norm=pi*np.power(2,(1-2*alpha),dtype=float)*sc.special.factorial(2*alpha)/sc.special.factorial(alpha)**2
        return F#/norm
    
    theta = np.linspace(0, 2*pi, 600)

    F=rad_pattern(ALPHA,0)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    p = ax.plot(theta, F,linewidth=3)
    plt.title(r'Radiation pattern for $\alpha$='+str(ALPHA),fontsize=20)

    power_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    power_slider = Slider(power_slider_ax, 'Alpha', 0, 30, valinit=0, valstep=1)
    
    param_slider_ax = plt.axes([0.2, 0.01, 0.6, 0.03])
    param_slider = Slider(param_slider_ax, 'Param', 0, pi, valinit=0, valstep=0.01)

    def update(val):
        power = power_slider.val
        param = param_slider.val
        F = rad_pattern(power,param)
        p[0].set_ydata(F)
        ax.set_ylim(0, max(F))
        fig.canvas.draw_idle()

    power_slider.on_changed(update)
    param_slider.on_changed(update)
    
    power_slider.label.set_size(20)
    param_slider.label.set_size(20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    #power_slider_ax.set_visible(False)
    #param_slider_ax.set_visible(False)
    #plt.savefig(f"../plots/radiation_pattern_alpha{ALPHA}.eps", format='eps', dpi=1000)
    return



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
    
    interactive_pattern()
    plot_radiation_pattern()

    
    
    
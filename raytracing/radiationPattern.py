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


if __name__=='__main__':
    plt.close("all")

    theta = np.linspace(0, 4*pi, 600)

    def rad_pattern(alpha,param):
        F = np.cos(theta/2+param)**alpha
        norm=pi*np.power(2,(2*alpha-1))*sc.special.factorial(2*alpha)/sc.special.factorial(alpha)**2
        return F/norm


    F=rad_pattern(2,0)
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111, projection='polar')
    p = ax.plot(theta, F)

    plt.title('Radiation Pattern',fontsize=20)

    power_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    power_slider = Slider(power_slider_ax, 'Alpha', 1, 30, valinit=2, valstep=1)
    
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

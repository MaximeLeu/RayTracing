#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:33:43 2023

@author: max
"""
import numpy as np
import scipy as sc
from scipy.constants import pi

from raytracing.materials_properties import P_IN,\
                                RADIATION_EFFICIENCY,\
                                RADIATION_POWER,\
                                ALPHA,\
                                TX_GAIN,\
                                RX_GAIN,\
                                FREQUENCY,\
                                LAMBDA,\
                                K,\
                                Z_0\


def vv_normalize(vv):
    norm=np.linalg.norm(vv)
    if norm == 0:#avoid dividing by 0
        return np.array([0,0,0])
    return vv/norm

def to_db(field_power):
    """
    converts given field power in watts to dB (normalised to INPUT_POWER)
    """
    #avoid computing log(0)
    P=np.atleast_1d(field_power)
    ind=np.where(P!=0)
    db=np.ones(len(P))*np.NINF
    for i in ind:
        db[i]=10*np.log10(P[i]/P_IN)
    return db[0] if np.isscalar(field_power) else db



def path_loss(d):
    """
    Given the distance between TX and RX antennas compute the path loss in dB
    """
    pr_pt=((LAMBDA/(4*pi*d))**2) *RX_GAIN*TX_GAIN*radiation_pattern(theta=0,phi=0)**2
    pl=10*np.log10(pr_pt)
    return pl


def radiation_pattern(theta,phi=0):
    F = np.cos(theta/2)**(2*ALPHA)
    norm=pi*np.power(2,(1-2*ALPHA),dtype=float)*sc.special.factorial(2*ALPHA)/(sc.special.factorial(ALPHA)**2)
    #print(f"theta {theta*180/pi:.2f}\u00b0 rad pattern={F:.2f}")
    return F#/norm
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:42:15 2023

@author: max
"""
import matplotlib.pyplot as plt
import numpy as np

L=50
ZTX=3
ZRX=30
ALPHA=np.pi/6

h=L*np.sin(ALPHA)
d=np.sqrt(L**2-h**2)

#ground
x_ground=[0,d]
y_ground=[0,-h]

#antennas
x_tx=[0,0]
y_tx=[0,ZTX]
x_rx=[d,d]
y_rx=[-h,-h+ZRX]

#los
x_los=[0,d]
y_los=[ZTX,-h+ZRX]

#reflection point
a=ZTX**2-ZRX**2
b=2*ZRX**2*L+2*ZRX**2*ZTX*np.sin(ALPHA)-2*ZTX**2*ZRX*np.sin(ALPHA)
c=-ZRX**2*L**2-2*ZRX**2*ZTX*L*np.sin(ALPHA)
determinant=b**2-4*a*c
assert(determinant>=0)
L2=(-b+np.sqrt(determinant))/(2*a)
assert(L2>0)

L1=L-L2
d1=np.sqrt(ZTX**2+L1**2+2*ZTX*L1*np.sin(ALPHA))
d2=np.sqrt(ZRX**2+L2**2-2*ZRX*L2*np.sin(ALPHA))
theta=np.arcsin(np.cos(ALPHA)*ZTX/d1)

#TX to G
d1_x=[0,L1*np.cos(ALPHA)]
d1_y=[ZTX,-L1*np.sin(ALPHA)]

#G to RX
d2_x=[L1*np.cos(ALPHA),d]
d2_y=[-L1*np.sin(ALPHA),-h+ZRX]

#other quantities not useful to draw the diagram
dlos=np.sqrt(d1**2+d2**2+2*d1*d2*np.cos(2*theta))
theta_tx=np.arcsin(d2/dlos*np.sin(2*theta))
theta_rx=np.arcsin(d1/dlos*np.sin(2*theta))
chi_tx=np.pi/2+ALPHA
chi_rx=np.pi/2-ALPHA
psi_tx=np.pi/2-ALPHA-theta
psi_rx=np.pi/2+ALPHA-theta

#plots
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

print(f"Given L={L:.2f}m, alpha={ALPHA*180/np.pi:.2f}\u00b0, ZTX={ZTX:.2f}m and ZRX={ZRX:.2f}m")
print(f"d={d:.2f}m, h={h:.2f}m, d1={d1:.2f}m, d2={d2:.2f}m, theta={theta*180/np.pi:.2f}\u00b0, theta_rx={theta_rx*180/np.pi:.2f}\u00b0, theta_tx={theta_tx*180/np.pi:.2f}\u00b0")

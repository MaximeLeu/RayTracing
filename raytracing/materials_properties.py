#pylint: disable=invalid-name,line-too-long
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:47:11 2022

@author: Maxime Leurquin

Properties of various materials
"""

import pandas as pd
from scipy.constants import epsilon_0, pi,c
#relative permittivity= epsilon
#relative permeability=mu
#electrical conductivity=sigma
#roughness describes the standard deviation of surface height in METERS

#TODO: find correct relatives mu for each materials

N_TREES=10 #how many trees to add to the place
MIN_TREE_HEIGHT=10
MAX_TREE_HEIGHT=20
TREE_SIZE=2
FREQUENCY=12.5*1e9 #in Hz

LAMBDA=c/FREQUENCY
K=2*pi/LAMBDA
Z_0=120*pi


P_IN=1
RADIATION_EFFICIENCY=1
RADIATION_POWER=RADIATION_EFFICIENCY*P_IN

ALPHA=10#increase alpha to make the antenna more directive.
#TX_GAIN=4*pi*1/((pi/6)**2) #4pi/(az*el), where az and el are the 3db beamdidths angles in radians
#RX_GAIN=4*pi*1/((pi/9)**2) #20 degree beamwidth  approx 103
#RX_GAIN=RX_GAIN*30 #*170 at 30GHz and *30 at 12.5GHz
TX_GAIN=450
RX_GAIN=300




ITU="ITU-R P.2040-2"



#Surface roughness values are expressed in meters
#mu,sigma and epsilon are given in relative values.
air={'material':"air",
 'epsilon':1,
 'mu':1,
 'sigma':0,
  'roughness': None,
 'frequency_range_GHz':"0.001-100",
 'source':ITU}

concrete={'material':"concrete",
 'epsilon':5.24,
 'mu':1,
 'sigma':0.0462*(FREQUENCY/1e9)**0.7822,
 'roughness': 1e-5,
 'frequency_range_GHz':"1-100",
 'source':ITU}

brick={'material':"brick",
 'epsilon':3.91,
 'mu':1,
 'sigma':0.0238*(FREQUENCY/1e9)**0.16,
 'roughness': 1e-4,
 'frequency_range_GHz':"1-40",
 'source':ITU}

wood={'material':"wood",
 'epsilon':1.99,
 'mu':1,
 'sigma':0.0047*(FREQUENCY/1e9)**1.0718,
 'roughness': 1e-4,
 'frequency_range_GHz':"0.001-100",
 'source':ITU}

glass={'material':"glass",
 'epsilon':6.31,
 'mu':1,
 'sigma':0.0036*(FREQUENCY/1e9)**1.3394,
 'roughness': 1e-7,
 'frequency_range_GHz':"0.1-100",
 'source':ITU}

metal={'material':"metal",
 'epsilon':1,
 'mu':1,
 'sigma':1e7,
 'roughness': 1e-7,
 'frequency_range_GHz':"1-100",
 'source':ITU}

medium_dry_ground={'material':"medium_dry_ground",
 'epsilon':15*(FREQUENCY/1e9)**(-0.1),
 'mu':1,
 'sigma':0.035*(FREQUENCY/1e9)**1.63,
 'roughness': 1e-3,
 'frequency_range_GHz':"1-10",
 'source':ITU}


DF_PROPERTIES=pd.concat([pd.DataFrame.from_records([air,concrete,brick,wood,glass,metal,medium_dry_ground])])

#complex effective relative permittivity, as defined in ITU-R P.2040-2
if not 'epsilon_eff' in DF_PROPERTIES.columns:
    #if required, the imaginary part of the relative permittivity  can be obtained:=17.98*sigma/f (see ITU), 17.98=1/(2pi epsilon_0 *1e9)
    DF_PROPERTIES['epsilon_eff']=DF_PROPERTIES["epsilon"]-1j*DF_PROPERTIES["sigma"]/(epsilon_0*2*pi*FREQUENCY)


def set_properties(building_type):

    def create_dict(material):
        mu=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["mu"].values[0])
        epsilon=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["epsilon"].values[0])
        sigma=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["sigma"].values[0])
        roughness=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["roughness"].values[0])
        epsilon_eff=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["epsilon_eff"].values[0])
        properties={"material":material, "mu":mu,"epsilon":epsilon,"sigma":sigma, "roughness":roughness, "epsilon_eff":epsilon_eff}
        return properties

    match building_type:
        case "office":
            properties=create_dict("glass")
        case "appartments":
            properties=create_dict("brick")
        case "garage":
            properties=create_dict("concrete")
        case "ground":
            properties=create_dict("medium_dry_ground")
        case "road":
            properties=create_dict("concrete")
        case "tree":
            properties=create_dict("wood")    
        
        case _:
            #if its none of the above
            print(f"Properties for the type {building_type} have not been specified, using concrete")
            properties=create_dict("concrete")
    return properties



#to include into latex
#pd.set_option("display.precision", 3)
#print(DF_PROPERTIES.to_latex(index=False))






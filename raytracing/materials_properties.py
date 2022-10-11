#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:47:11 2022

@author: maxime

Properties of various materials
"""

import pandas as pd
import numpy as np

#relative permittivity= epsilon
#relative permeability=mu
#electrical conductivity=sigma
#roughness describes the standard deviation of surface height in METERS

#TODO: find correct relatives mu for each materials

ITU="ITU-R P.2040-2"


#TODO find surface roughness values
#!!! I am stocking relative values
air={'material':"air",
 'epsilon':1,
 'mu':1,
 'sigma':0,
  'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"0.001-100",
 'source':ITU}

concrete={'material':"concrete",
 'epsilon':5.24+0j,
 'mu':1,
 'sigma':0.0462+0.7822j,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"1-100",
 'source':ITU}

brick={'material':"brick",
 'epsilon':3.91+0j,
 'mu':1,
 'sigma':0.0238+0.16j,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"1-40",
 'source':ITU}

wood={'material':"wood",
 'epsilon':1.99+0j,
 'mu':1,
 'sigma':0.0047+1.0718j,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"0.001-100",
 'source':ITU}

glass={'material':"glass",
 'epsilon':6.31+0j,
 'mu':1,
 'sigma':0.0036+1.3394j,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"0.1-100",
 'source':ITU}

metal={'material':"metal",
 'epsilon':1+0j,
 'mu':1,
 'sigma':10**7,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"1-100",
 'source':ITU}

medium_dry_ground={'material':"medium_dry_ground",
 'epsilon':15-0.1j,
 'mu':1,
 'sigma':0.035+1.63j,
 'roughness': 1 *10**(-6), #TODO find correct value
 'frequency_range_GHz':"1-10",
 'source':ITU}


DF_PROPERTIES=pd.concat([pd.DataFrame.from_records([air,concrete,brick,wood,glass,metal,medium_dry_ground])])



def set_properties(building_type):
  
    def create_dict(material):
        mu=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["mu"].values[0])
        epsilon=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["epsilon"].values[0])
        sigma=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["sigma"].values[0])
        roughness=str(DF_PROPERTIES.loc[DF_PROPERTIES["material"]==material]["roughness"].values[0])
        properties={"material":material, "mu":mu,"epsilon":epsilon,"sigma":sigma, "roughness":roughness}
        return properties
    
    match building_type:
        case "office":
            properties=create_dict("glass")
        case "appartments":
            properties=create_dict("brick")  
        case "garage":
            properties=create_dict("concrete")
        case "road":
            properties=create_dict("concrete")
        case _:
            #if its none of the above
            print(f"Properties for the type {building_type} have not been specified, using concrete")
            properties=create_dict("concrete")
    return properties




pd.set_option("display.precision", 3)
print(DF_PROPERTIES.to_latex(index=False))

#TODO add road/ground properties






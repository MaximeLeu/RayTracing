#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:02:07 2023

@author: Maxime Leurquin
"""

import numpy as np
import matplotlib.pyplot as plt

import electromagnetism_fun.electromagnetism as electromagnetism
import electromagnetism_fun.electromagnetism_utils as electromagnetism_utils

from electromagnetism_fun.electromagnetism import ElectromagneticField,Antenna,Reflection,Diffraction

from electromagnetism_fun.materials_properties import FREQUENCY

import electromagnetism_fun.multithread_solve as multithread_solve
import electromagnetism_fun.place_utils as place_utils
import raytracing.plot_utils as plot_utils
import raytracing.file_utils as file_utils




#ANTENNA TESTS------------------------------------------------------
def test_antenna_align_to_point(antenna,point):
    """
    please check if the tx antenna points correctly towards P1, 
    and that the TX antenna's axis are orthogonal
    """
    #plotting
    fig=plt.figure()
    fig.set_dpi(300)
    ax=fig.add_subplot(1,1,1,projection='3d')
    colors=['r','g','b']
    ax=plot_utils.plot_world_frame(ax, colors)
    
    ax=antenna.plot_antenna_frame(ax,colors)
    
    plot_utils.add_text_at_point_3d_ax(ax, np.array([0,0,0]), 'O')
    plot_utils.add_points_to_3d_ax(ax, antenna.position, color='r', s=20)
    plot_utils.add_text_at_point_3d_ax(ax, antenna.position, 'TX')
    
    plot_utils.add_points_to_3d_ax(ax, point, color='b', s=20)
    plot_utils.add_text_at_point_3d_ax(ax, point, 'P1')
    plot_utils.add_line_to_3d_ax(ax,np.array([point,antenna.position]),color='g',alpha=0.3)
    
    #ensure axis orthogonality
    plot_utils.ensure_axis_orthonormal(ax)
    ax.set_title("Ensure TX points to P1, and TX axis are perpendicular")
    plt.show()
    return



def test_pp_transform(antenna, point):
    """
    test antenna changement of reference frames
    !!Assumes the antenna is correctly aligned to the point!!
    """
    point_test1=antenna.pp_transform(antenna.position,"W2A")
    assert np.allclose(point_test1,np.array([0,0,0])),\
       f"ppW2A test 1 failed: is {point_test1} should be 000"
       
    point_test2=antenna.pp_transform(np.array([0,0,0]),"A2W")
    assert np.allclose(point_test2,antenna.position),\
       f"ppA2W test 2 failed: is {point_test2} should be {antenna.position}"
    
    point_relative_to_antenna=antenna.pp_transform(point,"W2A")
    dist=np.linalg.norm(antenna.position-point)
    should_be=np.array([0,0,dist])
    assert np.allclose(point_relative_to_antenna,should_be),\
        f"ppW2A test3 failed: is {point_relative_to_antenna}, should be: {should_be}"
    
    point_test3=antenna.pp_transform(should_be,"A2W")
    assert np.allclose(point_test3,point),\
        f"ppA2W test 4 failed: is {point_test3}, should be: {point}"
    return

    
def test_vv_transform(antenna):
    """
    If this test fails either the antenna basis is wrongly set,
    or vector conversion has failed
    """
    #The basis of the antenna's frame is well set:
    ex_ant=antenna.vv_transform(antenna.basis[0],"W2A")
    ey_ant=antenna.vv_transform(antenna.basis[1],"W2A")
    ez_ant=antenna.vv_transform(antenna.basis[2],"W2A")
    assert np.allclose(ex_ant,np.array([1,0,0])),\
        f'vvW2A is {ex_ant} should be 100'
    assert np.allclose(ey_ant,np.array([0,1,0])),\
        f'vvW2A is {ey_ant} should be 010'
    assert np.allclose(ez_ant,np.array([0,0,1])),\
        f'vvW2A is {ez_ant} should be 001'
    
    #test vv_A2W by doing it in the other way:  
    assert np.allclose(antenna.vv_transform(ex_ant,"A2W"),antenna.basis[0]),\
        f"vv_A2W is {antenna.vv_transform(ex_ant,'A2W')} should be {antenna.basis[0]}"
    assert np.allclose(antenna.vv_transform(ey_ant,"A2W"),antenna.basis[1]),\
        f"vv_A2W is {antenna.vv_transform(ey_ant,'A2W')} should be {antenna.basis[1]}"
    assert np.allclose(antenna.vv_transform(ez_ant,'A2W'),antenna.basis[2]),\
        f"vv_A2W is {antenna.vv_transform(ez_ant,'A2W')} should be {antenna.basis[2]}"
    return


def test_basis_orthonormality(antenna):
    #ensure the basis of the antenna is orthonormal
    assert np.isclose(np.dot(antenna.basis[0],antenna.basis[1]),0),"x is not perp to y"
    assert np.isclose(np.dot(antenna.basis[0],antenna.basis[2]),0),"x is not perp to z"
    assert np.isclose(np.dot(antenna.basis[1],antenna.basis[2]),0),"y is not perp to z"
    return

def test_double_conversion(antenna,point):
    """
    Check that converting a point/vector from world frame to the antenna frame
    and back from the antenna frame to the world frame, returns the same point
    """
    #given point is in world coordinates
    point_in_antenna=antenna.pp_transform(point,"W2A")
    point_in_world=antenna.pp_transform(point_in_antenna,"A2W")
    assert np.allclose(point,point_in_world),\
        "Point 2 antenna followed by antenna to world does not yield the original point:"\
            f"got {point_in_world} should have {point}"
        
    vv=point
    vv_in_antenna=antenna.vv_transform(vv,"W2A")
    vv_in_world=antenna.vv_transform(vv_in_antenna,"A2W")
    assert np.allclose(vv,vv_in_world),\
        "vect 2 antenna followed by antenna to world does not yield the original vect:"\
            f"got {vv_in_world} should have {vv}"
    return
            

def test_incident_angles(antenna):
    on_x=np.array([10,0,0])
    on_x=antenna.pp_transform(on_x,"A2W")
    
    on_y=np.array([0,10,0])
    on_y=antenna.pp_transform(on_y,"A2W")
    
    on_z=np.array([0,0,10])
    on_z=antenna.pp_transform(on_z,"A2W")
    
    r0, el0, az0 = antenna.incident_angles(on_x)
    assert np.isclose(np.degrees(el0), 90), \
        f"incident elevation when point is on x axis is: el={np.degrees(el0)} should be el=90"
    assert np.isclose(np.degrees(az0), 0), \
        f"incident azimuth when point is on x axis is: az={np.degrees(az0)} should be az=0"

    r1, el1, az1 = antenna.incident_angles(on_y)
    assert np.isclose(np.degrees(el1), 90), \
        f"incident elevation when point is on y axis is: el={np.degrees(el1)} should be el=90"
    assert np.isclose(np.degrees(az1), 90), \
        f"incident azimuth when point is on y axis is: az={np.degrees(az1)} should be az=90"

    r2, el2, az2 = antenna.incident_angles(on_z)
    assert np.isclose(np.degrees(el2), 0), \
        f"incident elevation when point is on z axis is: el={np.degrees(el2)} should be el=0"
    #the azimutal angle is not defined for a point on the z axis. We need to be careful when points sit there.
    #similarly for a point at the origin, the azimuthal and elevation angles are not defined. We need to be careful as well.    
    return



#REFLECTION TESTS----------------------------------------

def test_split_reflections_path():
    points = [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
    expected_output = [[[0,0,0],[1,1,1],[2,2,2]],[[1,1,1],[2,2,2],[3,3,3]]]
    assert Reflection.split_reflections_path(points) == expected_output
    
    # Test that the function returns a list of length 1 if the input list has exactly 3 elements
    points = [[0,0,0],[1,1,1],[2,2,2]]
    expected_output = [[[0,0,0],[1,1,1],[2,2,2]]]
    assert Reflection.split_reflections_path(points) == expected_output
    
    # Test that the function returns the correct output for an input with multiple sub-lists of length 3
    points = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    expected_output = [[[0,0,0],[1,1,1],[2,2,2]],[[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5]]]
    assert Reflection.split_reflections_path(points) == expected_output
    
    return
    




#PATH LOSS TEST--------------------------------------------

def run_small_simu(npoints=16,order=2):
    place,tx,geometry=place_utils.create_small_place(npoints)
    place_utils.plot_place(place, tx)
    solved_em_path,solved_rays_path= multithread_solve.multithread_solve_place(place=place,tx=tx,geometry=geometry,order=order)
    return tx,solved_em_path,solved_rays_path
    
def plot_small_vs_path_loss(tx,solved_em_path):
    """
    comparison of the fields obtained on the small place and path loss.
    """
    df=electromagnetism_utils.load_df(solved_em_path)
    npoints=len(df['rx_id'].unique())
    
    simu_y=np.zeros(npoints)
    simu_x=np.zeros(npoints)
    pl=np.zeros(npoints)
    for receiver in range(npoints):
        rx_df=df.loc[df['rx_id'] == receiver]
        rx_coord=rx_df["receiver"].values[0]
        d=np.linalg.norm(tx-rx_coord)
        pl[receiver]=ElectromagneticField.path_loss(d)
        simu_x[receiver]=d #distance TX-RX
        simu_y[receiver]=electromagnetism_utils.to_db(np.sum(rx_df["path_power"].values))

    #plots
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title(f'Comparison between path loss and model at {FREQUENCY/1e9} GHz', fontsize=20)
    ax.plot(simu_x,simu_y,color="orange",marker='o',label='simulation')
    ax.plot(simu_x,pl,marker='o',label='path loss')
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('RX-TX distance',fontsize=20)
    ax.set_ylabel('Received power [dB]',fontsize=20)
    ax.legend(fontsize=20)
    plt.show()
    #plt.savefig("../results/plots/small_place_vs_pathLoss.eps", format='eps', dpi=300,bbox_inches='tight')
    return


    


if __name__ == '__main__':
    file_utils.chdir_to_file_dir(__file__)
    plt.close("all") 
    #ANTENNA TESTS---------------------------------------------
    def run_antenna_class_tests(ntimes):
        for i in range(ntimes):
            #define an antenna and align it to a random point
            antenna_position=np.random.rand(3,)*5
            point = np.random.rand(3,)*10
            antenna=Antenna(position=antenna_position)
            antenna.align_antenna_to_point(point)
                
            #actual tests
            test_antenna_align_to_point(antenna,point)
            test_basis_orthonormality(antenna)
            test_pp_transform(antenna,point)
            test_vv_transform(antenna)
            test_double_conversion(antenna,point)
            test_incident_angles(antenna)
            print(f'Antenna tests ran with success ({ntimes} times)')
            return
            
    run_antenna_class_tests(ntimes=10)
    
    #REFLECTION TESTS---------------------------------
    #test_split_reflections_path()    
    
    
    # tx,solved_em_path,solved_rays_path=run_small_simu(npoints=16,order=2)
    # plot_small_vs_path_loss(tx,solved_em_path)
    
    
   
    
   
    
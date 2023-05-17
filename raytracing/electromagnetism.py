#pylint: disable=invalid-name,line-too-long
"""
@author: Maxime Leurquin
"""
#packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
from scipy.constants import c, mu_0, epsilon_0, pi

#self written imports
import raytracing.geometry as geom
from raytracing.materials_properties import P_IN,\
                                RADIATION_EFFICIENCY,\
                                RADIATION_POWER,\
                                ALPHA,\
                                TX_GAIN,\
                                RX_GAIN,\
                                FREQUENCY,\
                                LAMBDA,\
                                K,\
                                Z_0,\
                                DF_PROPERTIES,\
                                N_TREES,\
                                TREE_SIZE,\
                                ISOTROPIC_ANTENNA


import raytracing.electromagnetism_utils as electromagnetism_utils
from raytracing.electromagnetism_utils import vv_normalize,cot

import raytracing.place_utils as place_utils
import raytracing.plot_utils as plot_utils
import raytracing.file_utils as file_utils



pd.set_option('display.max_columns', 10)
pd.options.display.float_format = '{:.2g}'.format


#I consider the first medium is always air.
mu_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["mu"].values[0]*mu_0
sigma_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["sigma"].values[0]
epsilon_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["epsilon"].values[0]*epsilon_0


class Antenna:
    def __init__(self,position):
        self.position=position   #the position of the antenna
        self.basis=None      #the basis of the antenna. Z is the direction the antenna points towards.
        self.radiation_efficiency=1
        self.polarisation=None
        self.mat=None #change of basis matrix

    def __str__(self):
        string=f'Antenna: position: {self.position} '\
                f'orientation: {self.basis} '\
                f'polarisation: {self.polarisation} '\
                f'efficiency: {self.radiation_efficiency}'
        return string
                
   
    def align_antenna_to_point(self,point):
        """
        create a new reference frame in the antenna
        new_z=unit vector from antenna to point
        new_x=unit vector perp to new_z and parallel to ground (so perp to world_z)
        new_y=cross(new_x,new_z)
        """
        new_z=vv_normalize(point-self.position)
        new_x=vv_normalize(np.cross(new_z,np.array([0,0,1])))
        new_y= vv_normalize(np.cross(new_x,new_z))
        self.basis=[new_x,new_y,new_z]
        #change between vertical and horizontal polarisations here
        self.polarisation=new_x 

        x, y, z = np.eye(3)#world frame
        #matrix to change from the world reference frame to the antenna's
        mat=np.array([
                    [np.dot(new_x,x),np.dot(new_x,y),np.dot(new_x,z)],
                   [np.dot(new_y,x),np.dot(new_y,y),np.dot(new_y,z)],
                   [np.dot(new_z,x),np.dot(new_z,y),np.dot(new_z,z)],
                   ])

        self.mat=mat
        return

    def incident_angles(self,point):
        """
        given a point in world coordinates
        returns elevation and azimuth angles of incident point relative to the antenna
        """
        new_point=self.pp_transform(point,"W2A")
        r,el,az=electromagnetism_utils.to_spherical(new_point)
        return r,el,az
    

    def plot_antenna_frame(self,ax,colors):
        """
        plots the antenna reference frame on ax
        """
        antenna_basis=self.basis
        origin=self.position
        for i in range(3):
            plot_utils.plot_vec(ax,antenna_basis[i],colors[i],origin=origin)
        return ax

    def pp_transform(self, point, direction):
        """
        Change the reference frame of a point based on the given direction.
        if direction==A2W: from antenna-->world
        if direction==W2A: from world-->antenna
        """
        assert direction in ['A2W', 'W2A'], "direction must be either 'A2W' or 'W2A'"
        origin = self.position
        if direction == 'A2W':
            new_point=self.mat.T@point #rotate
            new_point +=origin #translate
        else:
            tr_point=point-origin #translate
            new_point=tr_point@(self.mat.T) #rotate
        return new_point

    def vv_transform(self,vector,direction):
        """
        Change the reference frame of a vector based on the given direction.
        if direction==A2W: from antenna-->world
        if direction==W2A: from world-->antenna
        """
        assert direction in ["A2W", "W2A"], "direction must be either 'A2W' or 'W2A'"
        new_vv = vector @ self.mat if direction == 'A2W' else vector@ (self.mat.T)
        return new_vv
    
    def path_W2A(self,path):
        """
        Change the reference frame of a path (list of points)
        from world-->antenna
        """
        return [self.pp_transform(point,"W2A") for point in path]

    @staticmethod
    def compute_Ae(theta_rx,phi_rx=0):
        #compute the antenna's effective aperture
        Ae=(LAMBDA**2/(4*pi))*RX_GAIN*ElectromagneticField.radiation_pattern(theta_rx,phi_rx)
        return Ae





class ElectromagneticField:
    @staticmethod
    def from_path(path,tx_antenna):
        """
        Given a direct path between TX and RX in world reference frame, 
        computes the electric field at RX using the far field approximation
        :param path: TX-RX, or TX-obstruction point, in that order, relative to the world frame
        :type path: list of two ndarrays of shape (3,)
        returns the electromagnetic field in the world reference frame
        """
        assert len(path)==2,'given path is strange'
        assert len(path[0])==3 and len(path[1])==3,"given path does not have the right shape"
        assert tx_antenna.basis is not None,"tx antenna is not aligned"
       
        r,theta,phi=tx_antenna.incident_angles(path[1])
        r_vv=tx_antenna.vv_transform(path[1]-path[0], "W2A")
        r_vv=vv_normalize(r_vv)

        #field in antenna's coordinates
        E=-1j*K*Z_0/(4*pi)*np.exp(-1j*K*r)/r*ElectromagneticField.radiation_pattern(theta,phi)
        E=E*vv_normalize(np.cross(np.cross(r_vv,tx_antenna.vv_transform(tx_antenna.polarisation,"W2A")),r_vv)) 
        E=tx_antenna.vv_transform(E,"A2W")#return to world coordinates
        return E

    @staticmethod
    def account_for_trees(E,points,place):
        """
        Attenuates the field if the path goes through a tree crown
        :param E: the electromagnetic field
        :type E: ndarray
        :param points: two points describing line path
        :type points: numpy.ndarray *shape=(2, 3)*
        """
        assert len(points)==2,"more than 2 points given"
        length=geom.tree_obstruction_length(place.get_polygons_iter(), points)

        if np.sum(length)>0:
            #when multiple trees are in the way
            print(f"{len(length)} trees in the way, attenuating, length: {length}")
            for d in length:
                #attenuation that adds to path loss
                L_db=0.39*((FREQUENCY/(1e6))**0.39)*d**0.25
                attenuation=10**(-L_db/10)
                print(f"attenuation: {L_db:.2f} dB, thus attenuation factor= {attenuation:.2f} ")
                E=E*attenuation
        return E

    @staticmethod
    def path_loss(d):
        """
        Given the distance between TX and RX antennas compute the path loss in dB
        """
        pr_pt=((LAMBDA/(4*pi*d))**2)*RX_GAIN*TX_GAIN*(ElectromagneticField.radiation_pattern(theta=0,phi=0))**2 
        pl=10*np.log10(pr_pt)
        return pl

    @staticmethod
    def radiation_pattern(theta,phi=0):
        #maximal when theta=0
        F = np.cos(theta/2)**(2*ALPHA)
        #norm=pi*np.power(2,(1-2*ALPHA),dtype=float)*sc.special.factorial(2*ALPHA)/(sc.special.factorial(ALPHA)**2)
        #print(f"theta {theta*180/pi:.2f}\u00b0 rad pattern={F:.2f}")
        return 1 if ISOTROPIC_ANTENNA else F
        #return F#/norm
    

class Reflection:
    @staticmethod
    def split_reflections_path(reflections_path):
        """
        :param reflections_path: a list representing a reflections path in the following format:[TX,R1,...RN,RX]
        :type reflections_path: list
        :return: a list of subpaths that includes the transmitter and receiver along with the reflections in between them
           for example [[TX,R1,P1],...,[PN,RN,RX]]
        :rtype: list 
        """
        assert len(reflections_path)>=3,f"wrong path shape given: {reflections_path}"
        points=reflections_path
        paths=[]
        for i in range(0,len(points)-2):
            paths.append(points[i:i+3])
        return paths

    @staticmethod
    def rayleigh_criterion(theta_i,roughness):
        cosi=np.cos(theta_i)
        rayleigh_criterion= (roughness > LAMBDA/(8*cosi))
        if rayleigh_criterion:
            print("applying scattering attenuation factor")
            chi=np.exp(-2*(K**2)*(roughness**2)*(cosi)**2)
        else:
            chi=1
        return chi

    @staticmethod
    def fresnel_coeffs(theta_i,polygon):
        """
        Computes the fresnel coefficients and applies the rayleigh roughness criterion
        :param theta_i: incident angle in radians
        :type theta_i: float
        :param polygon: the surface on which the fresnel coefficients are calculated
        :type polygon: OrientedPolygon
        :return: tuple with parallel and perpendicular fresnel coefficients
        :rtype: tuple of floats
        """
        assert 0<theta_i<pi/2, f"theta_i={theta_i*180/pi}\u00b0, but should between 0,90\u00b0"
        #assuming the first medium is air -> eps_eff_1=1
        roughness=float(polygon.properties['roughness'])
        epsilon_eff_2=np.complex(polygon.properties['epsilon_eff']) #relative value
        
        sqrt=np.sqrt(epsilon_eff_2-(np.sin(theta_i))**2)
        cosi=np.cos(theta_i)
        r_par=(epsilon_eff_2*cosi-sqrt) / (epsilon_eff_2*cosi+sqrt)    #Rh
        r_per=(cosi-sqrt) / (cosi+sqrt)    #Rs
        r_par=np.linalg.norm(r_par)
        r_per=np.linalg.norm(r_per)
        assert r_par<=1 and r_per<=1, f"fresnel coeffs are greater than 1: r_par {r_par} r_per {r_per}"

        chi=Reflection.rayleigh_criterion(theta_i,roughness)
        r_par*=chi
        r_per*=chi
        return r_par,r_per

    @staticmethod
    def dyadic_ref_coeff(path,reflection_polygon):
        """
        Computes the dyadic reflection coefficient of the path
            
        :param path: the points describing the path of the reflection
        :type path: np.array([TX,R,RX])
        :param reflection_polygon: the polygon on which the reflection occurs.
        :type reflection_polygon: OrientedPolygon
        
        :return: the dyadic reflection coefficient of the path, the norms of the vectors,
        the perpendicular polarization vector, the parallel polarization vector
        of the incident field, the parallel polarization vector of the reflected field
        :rtype: tuple(complex 3x3 matrix, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                      numpy.ndarray)
       """
        si,_,sr,_=geom.normalize_path(path) #normalized incident and reflected vectors
        surface_normal=reflection_polygon.get_normal()#always points outwards
        theta_i=np.arccos(np.dot(surface_normal,-si)) #incident angle
        assert(0<theta_i<pi/2),f'incident angle should be in 0,90\u00b0 but is: theta_i={theta_i*180/pi}\u00b0. Path is {path}, reflection polygon={reflection_polygon}'
        r_par,r_per=Reflection.fresnel_coeffs(theta_i,reflection_polygon)

        #dyadic reflection coeff McNamara 3.3, 3.4, 3.5, 3.6
        e_per=vv_normalize(np.cross(si, sr))
        ei_par= vv_normalize(np.cross(e_per,si))
        er_par= vv_normalize(np.cross(e_per,sr))

        R=e_per.reshape(3,1)@e_per.reshape(1,3)*r_per + ei_par.reshape(3,1)@er_par.reshape(1,3)*r_par #3x3 matrix, McNamara 3.39
        assert R.shape==(3,3)
        return R, e_per,ei_par, er_par
    
    
    @staticmethod
    def spread_factor(Si,Sr):
        #spread_factor=1 #for plane wave incidence
        spread_factor=Si/(Si+Sr) #for spherical wave incidence
        return spread_factor

    @staticmethod
    def my_reflect(E_i,reflections_path,surfaces_ids,rtp):
        """
        Compute the E field at the end of a reflection (reflection can be multiple) 
        
        :param E_i: the electric field at the first interaction point in V/m in the world reference frame
        :type E_i: numpy.ndarray shape=(3,)
        :param reflections_path:the points describing the path of the reflection
        :type reflections_path: numpy.ndarray shape=(N, 3)
        :param surfaces_ids: list of surfaces on which the path reflects
        :type surfaces_ids: list of int
        :param rtp: the RayTracingProblem containing all the geometry
        :type rtp: RayTracingProblem
        
        :return: the electric field at the end of the reflection path, in the world reference frame
        :rtype: numpy.ndarray shape=(3,)
        """ 
        paths=Reflection.split_reflections_path(reflections_path)
        E_r=E_i
        for ind, surf_id in enumerate(surfaces_ids):
            the_reflection_polygon=rtp.polygons[surf_id]
            the_path=paths[ind]
            _,Si,_,Sr=geom.normalize_path(the_path) #norms of incident and ref vectors
            R,e_per,ei_par,er_par=Reflection.dyadic_ref_coeff(the_path,the_reflection_polygon)

            #TODO check that the decomposition is correct
            #a=np.dot(E_r,e_per)*e_per+np.dot(E_r,ei_par)*ei_par
            #assert np.allclose(a,E_r), f" REF: decomposition failed: {a}, field: {E_r}"

            E_r=ElectromagneticField.account_for_trees(E_r,np.array([the_path[0],the_path[1]]),rtp.place) #account for TX-->ref # incident field at the point of reflection.
            
            field=(E_r.reshape(1,3)@R)*np.exp(-1j*K*Sr)*Reflection.spread_factor(Si,Sr)
            E_r=field.reshape(3,)
            assert np.linalg.norm(E_i)>=np.linalg.norm(field),\
                "reflected field should be less powerful than incident field: "\
                f"|E_i|={np.linalg.norm(E_i)} |E_r|={np.linalg.norm(field)} R={R}"
            
        last_path=paths[-1] #[Tx,R,RX]
        last_path=np.array([last_path[1],last_path[2]]) #[R,RX]
        E_r=ElectromagneticField.account_for_trees(E_r,last_path,rtp.place) #account for ref-->RX
        return E_r
    


class Diffraction:
    @staticmethod
    def F(x):
        #Exactly computes the transition function, as demonstrated in
        #https://eertmans.be/research/utd-transition-function/
        #F is defined in McNamara 4.72
        factor = np.sqrt(np.pi / 2)
        sqrtx = np.sqrt(x)
        S, C = sc.special.fresnel(sqrtx / factor)
        transition_function=2j * sqrtx* np.exp(1j * x)*(factor * ((1 - 1j) / 2 - C + 1j * S))
        return transition_function
    
    @staticmethod
    def distance_factor(Si,Sd,beta_0):
        #L is a distance parameter depending on the type of illumination.
        #For spherical wave incidence:
        L=((np.sin(beta_0))**2)*Si*Sd/(Si+Sd) #McNamara 6.26
        #For plane wave incidence: 
        #L=Si*(np.sin(beta_0))**2
        return L
    
    @staticmethod
    def spread_factor(Si,Sd):
        #for spherical wave incidence on a straight edge: 
        A=np.sqrt(Si/(Sd*(Sd+Si)))
        #for plane wave incidence:
        #A=1/np.sqrt(Sd)
        return A
    
    @staticmethod
    def A(x,n,sign):
    #definition is given in McNamara 4.66
        match sign:
            case "minus":
                N=round((-pi+x)/(2*n*pi))
            case "plus":
                N=round((pi+x)/(2*n*pi))
            case _:
                raise ValueError('sign should either be plus or minus')
        result=2*(np.cos(n*pi*N-x/2))**2
        return result

    def diff_dyadic_components(diff_path,polygon1,polygon2,e):
        """
        polygon1: the diffraction polygon: the 0-face.
        polygon2: the n-face
        e:the edge
        diff_path: the path the diffraction follows
        """
        si,Si,sd,Sd=geom.normalize_path(diff_path)#normalised incident and diff vectors, and their norms
        n_0=polygon1.get_normal() #unit vector normal to the 0 face
        n_1=polygon2.get_normal() #unit vector normal to the n face
        t_0=vv_normalize(np.cross(n_0,e)) #unit vector tangential to the O-face (McNamara 6.8) 
        alpha=pi-np.arccos(np.dot(n_0,n_1)) #angles between two planes = pi-angles between their outwards normals
        n=2-alpha/pi
        assert 0<alpha<pi and 1<=n<=2,\
            f"alpha= {alpha*180/pi}\u00b0, we restrict ourselves to wedges with interior angles in 0,180\u00b0"
        
        #beta_0 and beta_0' are the same (keller diffraction law)
        beta_0=np.arccos(np.dot(si,e)) #(scalar) angle between the incident ray and the edge (McNamara 4.19)
        assert 0<=beta_0<=pi #McNamara 4.20a

        #components of si and sd lying in the plane perpendicular to the edge (=s_t' and s_t in McNamara)
        si_t=vv_normalize(si-np.dot(si,e)*e) #(unit vector) McNamara 6.9
        sd_t=vv_normalize(sd-np.dot(sd,e)*e)#(unit vector) McNamara 6.10

        phi_i=pi-(pi-np.arccos(np.dot(-si_t,t_0)))*np.sign(np.dot(-si_t,n_0)) #(scalar) McNamara 6.11 incident angle with respect to the O-face
        phi_d=pi-(pi-np.arccos(np.dot(sd_t,t_0)))*np.sign(np.dot(sd_t,n_0)) #(scalar) McNamara 6.12 diffraction angle with respect to the O-face
        phi_i=phi_i%(2*pi)#get back in the 0-2pi range
        phi_d=phi_d%(2*pi)
        
        try:#text right after McNamara 6.12:
            assert 0<=phi_i<=n*pi , f"phi_i ={phi_i*180/pi}\u00b0 cannot be inside the wedge, alpha={(2-n)*pi*180/pi}\u00b0"
            assert 0<=phi_d<=n*pi, f"phi_d {phi_d*180/pi}\u00b0 cannot be inside the wedge, alpha={(2-n)*pi*180/pi}\u00b0"
        except AssertionError as e:
            raise e
        
        #components of the diffraction coefficients:
        common_factor=-np.exp(-1j*pi/4)/(2*n*np.sqrt(2*pi*K)*np.sin(beta_0)) #scalar
        L=Diffraction.distance_factor(Si,Sd,beta_0)

        cot1=cot((pi+(phi_d-phi_i))*1/(2*n)) #scalar
        cot2=cot((pi-(phi_d-phi_i))*1/(2*n))
        cot3=cot((pi+(phi_d+phi_i))*1/(2*n))
        cot4=cot((pi-(phi_d+phi_i))*1/(2*n))

        F1=Diffraction.F(K*L*Diffraction.A(phi_d-phi_i,n,"plus")) #scalar
        F2=Diffraction.F(K*L*Diffraction.A(phi_d-phi_i,n,"minus"))
        F3=Diffraction.F(K*L*Diffraction.A(phi_d+phi_i,n,"plus"))
        F4=Diffraction.F(K*L*Diffraction.A(phi_d+phi_i,n,"minus"))

        D1=common_factor*cot1*F1 #scalar McNamara 6.21
        D2=common_factor*cot2*F2 #scalar McNamara 6.22
        D3=common_factor*cot3*F3 #scalar McNamara 6.23
        D4=common_factor*cot4*F4 #scalar McNamara 6.24
        return D1,D2,D3,D4
   
        
    
    
    def my_diff(E_i,diff_path,diff_surfaces_ids,corner_points,rtp,receiver):
        """
        Compute the E field at the end of a diffraction in V/m
        returns the E field in world frame
        """
        si,Si,sd,Sd=geom.normalize_path(diff_path)#normalised incident and diff vectors, and their norms
        polygon_1 = rtp.polygons[diff_surfaces_ids[0]] #we define this as the 0-face.
        polygon_2 = rtp.polygons[diff_surfaces_ids[1]] #we define this as the n-face
        n_0=polygon_1.get_normal() #(vector) unit vector normal to the 0 face
        e= vv_normalize(np.diff(corner_points, axis=0).reshape(-1)) #edge

        #ensuring that e is such that t_0 points towards the 0 face (McNamara 6.8)
        t_0=vv_normalize(np.cross(n_0,e)) #(vector) unit vector tangential to the O-face (McNamara 6.8)
        mid_edge=geom.midpoint(corner_points[0], corner_points[1])
        if not polygon_1.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8):
            e=-e #flip edge orientation
            t_0=vv_normalize(np.cross(n_0,e))
        assert polygon_1.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8), "edge is wrongly oriented"

        try: #diffraction coefficients components:
            D1,D2,D3,D4=Diffraction.diff_dyadic_components(diff_path,polygon_1,polygon_2,e)
        except AssertionError:
            print("ERROR: some diffracted rays may be inside the geometry, the RX or TX may be inside a building.")
            print(f'Problematic diff path: {diff_path}')
            fig2 = plt.figure(f"ERROR: Rays reaching RX{receiver} are inside geometry",figsize=(8,5))
            fig2.set_dpi(300)
            ax = fig2.add_subplot(1, 1, 1, projection = '3d')
            rtp.plot3d(ax=ax,receivers_indexs=[receiver],legend=True,show_refl=False)
            raise
            
        r_per=1
        r_par=1
        #diffraction coefficients
        D_per=D1+D2+r_per*(D3+D4) #(scalar) McNamara 6.20
        D_par=D1+D2+r_par*(D3+D4) #(scalar) McNamara 6.20

        #Edge fixed coordinate system
        phi_i_vv=vv_normalize(np.cross(-e,si))#(vector) McNamara 6.2
        phi_d_vv=vv_normalize(np.cross(e,sd))#(vector) McNamara 6.4
        beta_0_i_vv=vv_normalize(np.cross(phi_i_vv,si))# (vector) McNamara 6.3
        beta_0_d_vv=vv_normalize(np.cross(phi_d_vv,sd))#(vector) McNamara 6.5

        #dyadic diffraction coefficient
        D=beta_0_i_vv.reshape(3,1)@beta_0_d_vv.reshape(1,3)*(-D_per) - phi_i_vv.reshape(3,1)@phi_d_vv.reshape(1,3)*D_par # (3x3 matrix) McNamara 6.19
        assert D.shape==(3,3), f"dyadic diffraction coefficient has wrong shape: D={D}"

        #field reaching receiver after a diffraction:
        field=(E_i.reshape(1,3)@D)*Diffraction.spread_factor(Si,Sd)*np.exp(-1j*K*Sd) #(vector) McNamara 6.18
        #diffracted field should be less powerful than the incident field
        assert np.linalg.norm(E_i)>=np.linalg.norm(field),\
            "diffracted field should be less powerful than incident field:"\
            f"|Ei|={np.linalg.norm(E_i)} and |Ed|= {np.linalg.norm(field)} dyadic coeff {D}"
        E_i=field.reshape(3,)
        E_i=ElectromagneticField.account_for_trees(E_i,np.array([diff_path[1],diff_path[2]]),rtp.place)
        return E_i
    
    
    
class SolvedField:
    """
    class to store the data about the field computed at the end of a path
    """
    def __init__(self,path):
        if path is not None:
            self.rx=path[-1] #1x3 numpy.ndarray
            self.tx=path[0]
            self.world_path=path #in world coordinates
            self.path_len=geom.path_length(path)
            self.time=self.path_len/c #float
        else:
            self.rx=None
            self.tx=None
            self.world_path=None
            self.path_len=None
            self.time=None
        

        self.path_type=None #string
        self.rx_id=None #int
        self.field=np.array([0,0,0]) #stored in world coordinates, the field incident at the RX antenna.
        
        self.rx_antenna=None
        self.tx_antenna=None
        #elevation and azimuth angles of the TX and RX rays, compared to the antenna axis
        self.tx_el=None
        self.tx_az=None
        self.rx_el=None
        self.rx_az=None

        self.power=0 #float


    def __str__(self):
        return f'{self.path_type} path from receiver {self.rx_id} with coordinates {self.rx} field strength {np.linalg.norm(self.field)}'

    def show_antennas_alignement(self):
        """
        shows the antennas, and their reference frames
        some vectors may not appear to be unit vectors, however they are:
        its just matplotlib scaling the axes.
        """
        fig=plt.figure()
        fig.set_dpi(300)
        ax=fig.add_subplot(1,1,1,projection='3d')
        colors=['r','g','b']
        ax=plot_utils.plot_world_frame(ax, colors)
        ax=plot_utils.plot_path(ax, self.world_path)
        ax=self.tx_antenna.plot_antenna_frame(ax, colors)
        ax=self.rx_antenna.plot_antenna_frame(ax,colors)
        ax.legend()
        plt.show()
        return

    def compute_rx_angles(self):
        penultimate_point=self.world_path[-2]
        #change of basis is done in antenna.incident_angles
        _,self.rx_el,self.rx_az=self.rx_antenna.incident_angles(penultimate_point)
        return

    def compute_tx_angles(self):
        second_point=self.world_path[1]
        _,self.tx_el,self.tx_az=self.tx_antenna.incident_angles(second_point)
        return

    def add_to_df(self,df):
        tx_angles=None
        rx_angles=None
        if self.tx_el is not None and self.tx_az is not None:
            tx_angles=np.array(np.degrees([self.tx_el,self.tx_az]))
        if self.rx_el is not None and self.rx_az is not None:
            rx_angles=np.array(np.degrees([self.rx_el,self.rx_az]))
        row = {
            'rx_id': self.rx_id,
            'receiver': self.rx,
            'path_type': self.path_type,
            'tx_angles': tx_angles,
            'rx_angles': rx_angles,
            'time_to_receiver': self.time,
            'field_strength': self.field,
            'path_power': self.power
        }
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return new_df
        
    @staticmethod
    def create_sol_df():
        #make sure the order of the columns is consistent with the one of add_to_df
        df=pd.DataFrame(columns=['rx_id',
                                 'receiver',
                                 'path_type',
                                 'tx_angles',
                                 'rx_angles',
                                 'time_to_receiver',
                                 'field_strength',
                                 'path_power'])
        return df

def find_shortest_path(rtp,receiver):
    """
    Goes through the rtp of the receiver to see which path is the shortest
    the antennas will then be aligned to this path for this receiver
    """
    if rtp.los[receiver]!=[]:
        print('the shortest path is a LOS')
        return rtp.los[receiver][0]
    path_type=None
    shortest_path=None
    shortest_path_lenght=np.inf
    for order in rtp.diffractions[receiver]:
        for path in range(0,len(rtp.diffractions[receiver][order])):
            the_data=rtp.diffractions[receiver][order][path]
            the_path=the_data[0]
            path_len=geom.path_length(the_path)
            if path_len<shortest_path_lenght:
                shortest_path_lenght=path_len
                shortest_path=the_path
                path_type="Diff"

    for order in rtp.reflections[receiver]:
        for path in range(0,len(rtp.reflections[receiver][order])):
            the_data=rtp.reflections[receiver][order][path]
            the_path=the_data[0]
            path_len=geom.path_length(the_path)
            if path_len<shortest_path_lenght:
                shortest_path_lenght=path_len
                shortest_path=the_path
                path_type="Ref"

    print(f'the shortest path is a {path_type}')
    if path_type is None:
        print(f"No path was found between TX and RX{receiver}")
    return shortest_path



def my_field_computation(rtp,save_name="problem_fields"):
    """
    For each receiver compute the field of all of its paths
    """
    tx=rtp.emitter[0]
    print(f"TX {tx}")
    #Computes the resulting field at the receiver from all reflections and diffractions in V/m
    def compute_LOS(receiver,rtp,solved_list,tx_antenna):
        """
        compute field from LOS for the given receiver
        append the result to solved_list
        """
        print('Computing LOS field')
        #receiver is the index of the receiver NOT its coordinates
        if rtp.los[receiver]!=[]:
            sol=SolvedField(rtp.los[receiver][0])
            sol.rx_id=receiver
            sol.path_type="LOS"
            sol.tx_antenna=tx_antenna
            sol.compute_tx_angles()
            E_los=ElectromagneticField.from_path(sol.world_path,sol.tx_antenna)
            E_los=ElectromagneticField.account_for_trees(E_los,sol.world_path,rtp.place)
            sol.field=E_los
            solved_list.append(sol)
        else:
            print(f'NO LOS for RX{receiver}')
        return

    def compute_reflections(receiver,rtp,solved_list,tx_antenna):
        """
        For each pure reflections path reaching this receiver compute the field
        append the result to solved_list
        """
        print('Computing REF fields')
        reflections = rtp.reflections
        for order in reflections[receiver]:
            for path in range(0,len(reflections[receiver][order])):
                the_data=reflections[receiver][order][path]
                the_path=the_data[0]
                the_reflection_surfaces_ids=the_data[1]
                first_interaction=the_path[1] #second point of the path

                sol=SolvedField(the_path)
                sol.rx_id=receiver
                sol.path_type='R'*order
                sol.tx_antenna=tx_antenna
                sol.compute_tx_angles()

                E_i=ElectromagneticField.from_path([sol.tx,first_interaction],sol.tx_antenna)
                this_E=Reflection.my_reflect(E_i,sol.world_path,the_reflection_surfaces_ids,rtp)

                sol.field=this_E
                solved_list.append(sol)
        return

    def compute_diffractions(receiver,rtp,solved_list,tx_antenna):
        """
        For each path containing a diffraction that reaches this receiver
        compute the field and append the result to solved_list
        """
        print('Computing DIFF fields')
        diffractions = rtp.diffractions
        for order in diffractions[receiver]:
            for path in range(0,len(diffractions[receiver][order])):
                the_data=diffractions[receiver][order][path]
                the_path=the_data[0]
                the_reflection_surfaces_ids=the_data[1]
                the_diff_surfaces_ids=the_data[2]
                the_corner_points=the_data[3]
                #WARNING: assumes diff always happen last
                the_path_before_diff=the_path[0:-1]
                the_diff_path=the_path[-3:] #last 3 points
                first_interaction=the_path[1] #second point of the path

                E_0=ElectromagneticField.from_path([tx,first_interaction],tx_antenna)
                assert len(the_path)>=3, "path doesn't have the right shape"
                if len(the_path)==3:#TX-diff-RX
                    E_i=ElectromagneticField.account_for_trees(E_0,np.array([the_path[0],the_path[1]]),rtp.place)
                else:
                    #account for reflections happening before the diffraction
                    #trees are accounted for directly in my_reflect()
                    E_i=Reflection.my_reflect(E_0,the_path_before_diff,the_reflection_surfaces_ids,rtp) #E reaching the diffraction point

                #account for the diffraction
                this_E=Diffraction.my_diff(E_i,the_diff_path,the_diff_surfaces_ids,the_corner_points,rtp,receiver)

                sol=SolvedField(the_path)
                sol.rx_id=receiver
                sol.path_type='R'*(order-1)+'D'
                sol.tx_antenna=tx_antenna
                sol.compute_tx_angles()
                sol.field=this_E
                solved_list.append(sol)
        return


    #Add trees foliage on top of each tree trunk
    rtp.place.add_tree_crowns()
    #place_utils.plot_place(rtp.place,tx,name="place with tree crowns")
    
    #randomly add tree foliage everywhere
    #for _ in range(0,N_TREES):
        #rtp.place.add_tree(tree_size=TREE_SIZE)
   

    #compute everything
    df=SolvedField.create_sol_df()
    
    #if true: align TX antenna to a specific point and let it be
    #rx antenna is always aligned in the best possible way
    align_once=False
    if align_once:
        align_point=rtp.receivers[40]
        print(f"ALIGNING TX ANTENNA TO {align_point} only ONCE")
    
    for receiver in range(0,len(rtp.solved_receivers)):
        print(f"\n EM SOLVING RECEIVER {receiver}")
        tx_antenna=Antenna(tx)
        rx_antenna=Antenna(rtp.receivers[receiver])

        best_path=find_shortest_path(rtp,receiver)
        if best_path is not None:
            tx_antenna.align_antenna_to_point(best_path[1])#first point
            rx_antenna.align_antenna_to_point(best_path[-2])#second to last point
            if align_once:
                tx_antenna.align_antenna_to_point(align_point)
           
            this_solved_list=[] #list containing the fields for all the path between tx and this rx
            compute_LOS(receiver,rtp,this_solved_list,tx_antenna)
            compute_reflections(receiver, rtp,this_solved_list,tx_antenna)
            compute_diffractions(receiver,rtp,this_solved_list,tx_antenna)
            for sol in this_solved_list:
                sol.rx_antenna=rx_antenna
                sol.compute_rx_angles()
                Ae=Antenna.compute_Ae(sol.rx_el,sol.rx_az)#rx radiation pattern and rx gain taken into account here
                field_polarisation=vv_normalize(rx_antenna.vv_transform(sol.field,"W2A"))
                polarisation_efficiency=(np.linalg.norm(np.dot(field_polarisation,rx_antenna.vv_transform(rx_antenna.polarisation,"W2A"))))**2
                print(f"polarisation_eff {polarisation_efficiency}")
                sol.power=1/(2*Z_0)*Ae*(np.linalg.norm(np.real(rx_antenna.vv_transform(sol.field,"W2A"))))**2
                sol.power=sol.power*polarisation_efficiency*TX_GAIN 
                #sol.show_antennas_alignement()
                print(f"rx {receiver} path {sol.path_type}: RX angles theta {sol.rx_el*180/pi:.2f}\u00b0 phi {sol.rx_az*180/pi:.2f}\u00b0")
                df=sol.add_to_df(df)

        else:#no path between the receiver and the tx
            this_solved_list=[]
            sol=SolvedField(path=None)
            sol.path_type="Impossible" #no path
            sol.rx=rtp.receivers[receiver]
            sol.rx_id=receiver
            sol.tx=tx
           
            df=sol.add_to_df(df)
    electromagnetism_utils.save_df(df,save_name)#saves into the result folder
    return df


if __name__ == "__main__":
    file_utils.chdir_to_file_dir(__file__)

















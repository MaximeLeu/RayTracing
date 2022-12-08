#self written imports
from raytracing import array_utils
from raytracing import geometry as geom
from materials_properties import DF_PROPERTIES, FREQUENCY,N_TREES,TREE_SIZE
import file_utils
import plot_utils



#packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
from scipy.constants import c, mu_0, epsilon_0, pi
pd.set_option('display.max_columns', 10)
pd.options.display.float_format = '{:.2g}'.format


#I consider the first medium is always air.
mu_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["mu"].values[0]*mu_0
sigma_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["sigma"].values[0]
epsilon_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["epsilon"].values[0]*epsilon_0
Z_0=120*pi
lam=c/FREQUENCY


P_IN=1
RADIATION_EFFICIENCY=1
RADIATION_POWER=RADIATION_EFFICIENCY*P_IN


TX_GAIN=4*pi*1/((pi/6)**2) #4pi/(az*el), where az and el are the 3db beamdidths angles in radians 
RX_GAIN=4*pi*1/((pi/9)**2) #20 degree beamwidth 

def vv_normalize(vv):
    norm=np.linalg.norm(vv)
    if norm == 0:#avoid dividing by 0
        return np.array([0,0,0])
    else:
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



class Antenna:
    def __init__(self):
        self.position=None   #the position of the antenna
        self.basis=None      #the basis of the antenna. Z is the direction the antenna points towards.
        self.radiation_efficiency=1
        self.E0=None         #the field radiated 1m away from the antenna 
        self.polarisation=None 
        
    def __str__(self):
        return f'Antenna: position: {self.position} eff: {self.radiation_efficiency} E0={self.E0} gain={self.gain}'
    
    def parametric_from_normal_and_point(normal,point):
        """
        Returns the parametric equation of the plane described by its normal and a point.
        It will return all four coefficients such that: a*x + b*y + c*z + d = 0.
        """
        normal=vv_normalize(normal)
        a, b, c = normal
        d=-np.dot(point, normal)
        return a,b,c,d    
    
    def align_antenna_to_point(self,point):
        """
        create a new reference frame in the antenna
        new_z=unit vector from antenna to point
        new_x=unit vector perp to new_z and parallel to ground (so perp to z)
        new_y=cross(new_x,new_z)
        """
        new_z=vv_normalize(point-self.position)
        new_x=vv_normalize(np.cross(new_z,np.array([0,0,1])))
        new_y= vv_normalize(np.cross(new_x,new_z))
        self.basis=[new_x,new_y,new_z]
        self.polarisation=new_y
        self.test_basis()
        return
    
    
    def test_basis(self):
        #ensure the basis is orthonormal
        assert np.isclose(np.dot(self.basis[0],self.basis[1]),0),"x is not perp to y"
        assert np.isclose(np.dot(self.basis[0],self.basis[2]),0),"x is not perp to z"
        assert np.isclose(np.dot(self.basis[1],self.basis[2]),0),"y is not perp to z"
        return

    def incident_angles(self,point):
        """
        given the antenna basis and the incoming ray 
        returns elevation and azimuth angles of incident path in radians 
        """
        new_point=self.change_reference_frame(point)
        r,el,az=Antenna.to_spherical(new_point)
        return r,el,az

    def change_reference_frame(self,point):
        """
        transform the given point into the reference frame of the antenna
        """
        origin=self.position
        x=np.array([1,0,0])
        y=np.array([0,1,0])
        z=np.array([0,0,1])
        
        new_x=self.basis[0]
        new_y=self.basis[1]
        new_z=self.basis[2]
        assert(np.isclose([np.linalg.norm(new_x),np.linalg.norm(new_y),np.linalg.norm(new_z)],[1,1,1])).all(),\
            f'x: {np.linalg.norm(new_x)} y {np.linalg.norm(new_y)} z {np.linalg.norm(new_z)}'
        
        mat=np.array([
                    [np.dot(new_x,x),np.dot(new_x,y),np.dot(new_x,z)],
                   [np.dot(new_y,x),np.dot(new_y,y),np.dot(new_y,z)],
                   [np.dot(new_z,x),np.dot(new_z,y),np.dot(new_z,z)],
                   ])
    
        tr_point=point-origin #translate
        new_point=tr_point@mat.T #rotate
        return new_point
    
    def test_change_reference_frame(tx,rx):
        #ensure tx and rx are well transformed.
        tx_ant=Antenna()
        tx_ant.position=tx
        tx_ant.align_antenna_to_point(rx)
        new_rx=tx_ant.change_reference_frame(rx)
        new_tx=tx_ant.change_reference_frame(tx)
        length=np.linalg.norm(tx-rx)
        new_length=np.linalg.norm(new_tx-new_rx)
        assert np.isclose(new_tx, np.array([0,0,0])).all(), f'new tx {new_tx}'
        assert np.isclose(new_length,length).all(),f'new length {new_length} length {length}'
        assert np.isclose(new_rx,np.array([0,0,length])).all()
        return
    
    def to_spherical(point):
       """
       transform the given point into spherical coordinate relative to the antenna's basis
       """
       #TODO: careful with the definitino of theta, maybe theta=pi/2-theta!!! 
       #for the moment theta=from z to y (remember z is the direction of the path)
       x, y, z = point.T
       hxy = np.hypot(x, y)
       r = np.hypot(hxy, z)
       el = np.arctan2(hxy, z)
       az = np.arctan2(y, x)
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
    
    def path_to_antenna_reference_frame(self,path):
        new_path=[]
        for point in path:
            new_point=self.change_reference_frame(point)
            new_path.append(new_point)        
        return new_path
    
    def antenna_to_world(self,point):

        origin=self.position
        x=np.array([1,0,0])
        y=np.array([0,1,0])
        z=np.array([0,0,1])
        
        new_x=self.basis[0]
        new_y=self.basis[1]
        new_z=self.basis[2]
        
        mat=np.array([
                    [np.dot(new_x,x),np.dot(new_x,y),np.dot(new_x,z)],
                   [np.dot(new_y,x),np.dot(new_y,y),np.dot(new_y,z)],
                   [np.dot(new_z,x),np.dot(new_z,y),np.dot(new_z,z)],
                   ])
        new_point=mat.T@point
        new_point= new_point+origin           
        
        
        return new_point
 
    def test_trans(self,point):
        antenna_point=self.change_reference_frame(point)
        world_point=self.antenna_to_world(antenna_point)
        assert(np.isclose(point,world_point)).all(), f'real: {point} invreted {world_point}'
        print(f'real: {point} mid {antenna_point} invreted {world_point}')
        return
    
     
 
   
        
def radiation_pattern(theta,phi):  
    F=np.sin(theta) #scalar
    F=abs(F)
    print(f"rad pattern {F}")
    return F 
    
class ElectromagneticField:

    def __init__(self, E=np.array([1, 0, 0], dtype=complex), f=FREQUENCY, v=c):
        self.E = E
        self.f = f
        self.v = v
        self.k = 2 * pi * f / v
        self.w = 2 * pi * f
        self.lam =v/f 


    def compute_E0(path,tx_antenna):
        """
        Computes the field radiated 1m away from the antenna in the direction of the path
        """
        r,phi,theta=tx_antenna.incident_angles(path[1]) 
        #transform r_VV into the basis of the antenna!!!
        new_point=tx_antenna.change_reference_frame(path[1])
        r_vv=vv_normalize(new_point) #tx_antenna.position=[0,0,0]
        Antenna.test_change_reference_frame(path[0],path[-1]) #small test
        F=np.sin(theta)*tx_antenna.basis[1]

        #field in antenna's coordinates
        EF_0=ElectromagneticField()
        E0=-1j*EF_0.k*Z_0*1/(4*pi*r)*np.exp(-1j*EF_0.k*r)*np.cross(np.cross(r_vv,F),r_vv)*np.sqrt(2*Z_0*TX_GAIN*P_IN/(4*pi))
        E0=E0*np.sqrt(radiation_pattern(theta, phi))
        #assert(np.dot(E0,tx_antenna.basis[0])),'ERROR: field is not transverse'
        
        
        #return to world coordinates
        E0=tx_antenna.antenna_to_world(E0)
        EF_0.E = E0 
        return EF_0

    @staticmethod
    def from_path(path,tx_antenna):
        """
        Given a direct path between TX and RX, computes the electric field at RX using the far field approximation  
        :param path: TX point and point of the first obstruction (or RX if line of sight)
        :type path: numpy.ndarray(shapely.geometry.Points) *shape=(2)
        """
        assert len(path[0])==3 and len(path[1])==3,"given path does not have the right shape"
        E0=ElectromagneticField.compute_E0(path,tx_antenna)
        return E0

    def fresnel_coeffs(E,theta_i,roughness,epsilon_eff_2):
        assert theta_i>0 and theta_i<pi/2, f"theta_i={theta_i*180/pi} degrees, but should between 0,90"
        #assuming the first medium is air
        sqrt=np.sqrt(epsilon_eff_2-(np.sin(theta_i))**2)
        cosi=np.cos(theta_i)
        r_par=(epsilon_eff_2*cosi-sqrt) / (epsilon_eff_2*cosi+sqrt)    #Rh 
        r_per=(cosi-sqrt) / (cosi+sqrt)    #Rs 
        r_par=np.linalg.norm(r_par)
        r_per=np.linalg.norm(r_per)
        assert r_par<=1 and r_per<=1, f"fresnel coeffs are greater than 1: r_par {r_par} r_per {r_per}"
        
        rayleigh_criterion= (roughness > E.lam/(8*cosi))
        if rayleigh_criterion:
            print("applying scattering attenuation factor")
            chi=np.exp(-2*(E.k**2)*(roughness**2)*(cosi)**2)
            r_par=r_par*chi
            r_per=r_per*chi
        return r_par,r_per
    
    
    def account_for_trees(E,points,place):
        """
        Attenuates the field if the path goes through a tree
        :param E: electromagnetic field object
        :type E: ElectromagneticField
        :param points: two points describing line path
        :type points: numpy.ndarray *shape=(2, 3)*
        """
        length=geom.tree_obstruction_length(place.get_polygons_iter(), points)
       
        if np.sum(length)>0: 
            #when multiple trees are in the way
            print(f"{len(length)} trees in the way, attenuating, length: {length}")
            for d in length:
                #attenuation that adds to path loss
                L_db=0.39*((FREQUENCY/10**6)**0.39)*d**0.25
                #attenuation=1/(10**(L_db/10))
                attenuation=1 #TODO
                print(f"attenuation: {L_db} dB, thus attenuation factor= {attenuation} ")
                E.E=E.E*attenuation     
        return E
    
       
    
    def get_parameters(reflection_polygon):    
        mu_2=np.complex(reflection_polygon.properties['mu'])*mu_0
        sigma_2=np.complex(reflection_polygon.properties['sigma'])
        epsilon_2=float(reflection_polygon.properties['epsilon'])*epsilon_0
        roughness=float(reflection_polygon.properties['roughness']) 
        epsilon_eff_2=np.complex(reflection_polygon.properties['epsilon_eff']) #relative value
        
        surface_normal=array_utils.normalize(reflection_polygon.get_normal())
        return mu_2,sigma_2,epsilon_2,epsilon_eff_2,roughness,surface_normal
    
    def my_reflect(E_i,reflections_path,surfaces_ids,rtp):
        """
        Compute the E field at the end of a reflection (reflection can be multiple) in V/m
        E_i is the field at the first reflection point
        """
        def split_reflections_path(reflections_path):
            """
            given [TX,R1,...RN,RX]
            Returns [[TX,R1,P1],...,[PN,RN,RX]]
            """  
            points=reflections_path    
            paths=list()
            for i in range(0,len(points)-2):
                paths.append(points[i:i+3])
            return paths
        
        def dyadic_ref_coeff(path,reflection_polygon):
            """   
            Computes the dyadic reflection coefficient of the path
            Parameters
            ----------
            path : TYPE [[TX,R1,RX],[R1 surface id]]
                DESCRIPTION. The points describing the path of the reflection and 
                the id of the surface on which the reflection took place

            polygons :problem.polygons
            The polygons of the ray tracing problem
            Returns
            -------
            R : TYPE complex 3x3 matrix
                DESCRIPTION the dyadic reflection coefficient of the path
           """
            #data:        
            mu_2,sigma_2,epsilon_2,epsilon_eff_2,roughness,surface_normal=ElectromagneticField.get_parameters(reflection_polygon)
            #Compute vectors from emitter to reflection point,
            #then from reflection point to receiver
            vectors = np.diff(path, axis=0)
            vectors, norms = geom.normalize_path(path)
            si = vectors[0, :] #normalised incident vector
            sr = vectors[1, :] #normalised reflected vector
            #incident angle and transmission angle
            theta_i=np.arccos(np.dot(-surface_normal,si)) #incident angle
            assert(theta_i<pi/2),f'strange incident angle: theta_i= {theta_i*180/pi} degrees'
            r_par,r_per=ElectromagneticField.fresnel_coeffs(E_i,theta_i,roughness,epsilon_eff_2)
            
            #dyadic reflection coeff McNamara 3.3, 3.4, 3.5, 3.6
            e_per=vv_normalize(np.cross(si, sr))
            ei_par= vv_normalize(np.cross(e_per,si))
            er_par= vv_normalize(np.cross(e_per,sr))
            
            #check that decomposition is correct
            #a=np.dot(E_i.E,e_per)*e_per+np.dot(E_i.E,ei_par)*ei_par
            #assert(a==E_i.E).all(),f"A: {a}, B: {E_i.E}"
            
            R=e_per.reshape(3,1)@e_per.reshape(1,3)*r_per + ei_par.reshape(3,1)@er_par.reshape(1,3)*r_par #3x3 matrix, McNamara 3.39
            assert(R.shape==(3,3))
            return R,norms, e_per,ei_par, er_par,r_par,r_per
        
        E_r=ElectromagneticField()
        E_r.E=E_i.E
        paths=split_reflections_path(reflections_path)
        for i in range(0,len(surfaces_ids)):
            reflection_polygon=rtp.polygons[surfaces_ids[i]]
            #R,norms,_,_,_,_,_=dyadic_ref_coeff(paths[i],reflection_polygon)
            R,norms,e_per,ei_par, er_par,r_par,r_per=dyadic_ref_coeff(paths[i],reflection_polygon)
            
            Si= norms[0]#norm of incident ray
            Sr= norms[1]#norm of reflected ray  
            E_r=ElectromagneticField.account_for_trees(E_r,np.array([paths[i][0],paths[i][1]]),rtp.place) #account for TX-->ref # incident field at the point of reflection.
            spread_factor=Si/(Si+Sr)
            field=(E_r.E.reshape(1,3)@R)*np.exp(-1j*E_r.k*Sr)*spread_factor
            #reflected field should be of lesser intensity than the incident field
            assert np.linalg.norm(E_i.E)>=np.linalg.norm(field), f"incident field strength {np.linalg.norm(E_i.E)} reflected field strength {np.linalg.norm(field)} dyadic coeff {R}"
            E_r.E=field.reshape(3,)          
        last_path=paths[-1] #[Tx,R,RX]
        last_path=np.array([last_path[1],last_path[2]]) #[R,RX]
        E_r=ElectromagneticField.account_for_trees(E_r,last_path,rtp.place) #account for ref-->RX
        return E_r


    def my_diff(E_i,diff_path,diff_surfaces_ids,corner_points,rtp,receiver):
        """
        Compute the E field at the end of a diffraction in V/m
        """
        def diff_dyadic_components(Si,si,Sd,sd,e,k,diffraction_polygon,n):
            """
            Computes the 4 components of the dyadic diffraction coefficient
            e: unit vector tangential to the edge
            si=s' in McNamara= incident ray
            sd=s in McNamara= diffracted ray
            """   
            def F(x):
                #Exactly computes the transition function, as demonstrated in
                #https://eertmans.be/research/utd-transition-function/
                #F is defined in McNamara 4.72
                factor = np.sqrt(np.pi / 2)
                sqrtx = np.sqrt(x)
                S, C = sc.special.fresnel(sqrtx / factor)
                transition_function=2j * sqrtx* np.exp(1j * x)* (factor * ((1 - 1j) / 2 - C + 1j * S))
                return transition_function
            
            def A(x,sign):
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
            
            def cot(x):
                return 1/np.tan(x)
                        
            def distance_factor(Si,Sd,beta_0):
                #L is a distance parameter depending on the type of illumination. 
                #For spherical wave incidence :
                L=((np.sin(beta_0))**2)*Si*Sd/(Si+Sd) #McNamara 6.26
                return L
            
            #beta_0 and beta_0' are the same (keller diffraction law)
            beta_0=np.arccos(np.dot(si,e)) #(scalar) angle between the incident ray and the edge (McNamara 4.19)
            assert(beta_0<=pi and beta_0>=0) #McNamara 4.20a
            
            n_0=array_utils.normalize(diffraction_polygon.get_normal()) #(vector) unit vector normal to the 0-face
            t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (McNamara 6.8)

            #unit vector components of si and sd lying in the plane perpendicular to the edge (=s_t' and s_t in McNamara)
            si_t=vv_normalize(si-np.dot(si,e)*e) #(vector) McNamara 6.9
            sd_t=vv_normalize(sd-np.dot(sd,e)*e)#(vector) McNamara 6.10
            
            
            phi_i=pi-(pi-np.arccos(np.dot(-si_t,t_0)))*np.sign(np.dot(-si_t,n_0)) #(scalar) McNamara 6.11 incident angle with respect to the O-face 
            phi_d=pi-(pi-np.arccos(np.dot(sd_t,t_0)))*np.sign(np.dot(sd_t,n_0)) #(scalar) McNamara 6.12 diffraction angle with respect to the O-face
            phi_i=phi_i%(2*pi)#get back in the 0-2pi range
            phi_d=phi_d%(2*pi)

            # text right after Mc 6.12 for these asserts:
            try:
                assert phi_i>=0 and phi_i<=n*pi , f"phi_i ={phi_i*180/pi} cannot be inside the wedge, alpha={(2-n)*pi*180/pi}"
                assert phi_d>=0 and phi_d<=n*pi, f"phi_d {phi_d*180/pi} cannot be inside the wedge, alpha={(2-n)*pi*180/pi}"      
            except AssertionError:
                print("ERROR: some diffracted rays may be inside the geometry, the receiver may be inside a building.")
                fig2 = plt.figure(f"ERROR: Rays reaching RX{receiver} inside geometry",figsize=(8,5))
                fig2.set_dpi(300)
                ax = fig2.add_subplot(1, 1, 1, projection = '3d')
                rtp.plot3d(ax=ax,receivers_indexs=[receiver],legend=True,show_refl=False) 
                raise
            #components of the diffraction coefficients:
            common_factor=-np.exp(-1j*pi/4)/(2*n*np.sqrt(2*pi*k)*np.sin(beta_0)) #scalar
            L=distance_factor(Si,Sd,beta_0)
        
            cot1=cot((pi+(phi_d-phi_i))*1/(2*n)) #scalar
            cot2=cot((pi-(phi_d-phi_i))*1/(2*n))
            cot3=cot((pi+(phi_d+phi_i))*1/(2*n))
            cot4=cot((pi-(phi_d+phi_i))*1/(2*n))
            
            F1=F(k*L*A(phi_d-phi_i,"plus")) #scalar
            F2=F(k*L*A(phi_d-phi_i,"minus"))
            F3=F(k*L*A(phi_d+phi_i,"plus"))
            F4=F(k*L*A(phi_d+phi_i,"minus"))
            
            D1=common_factor*cot1*F1 #scalar McNamara 6.21
            D2=common_factor*cot2*F2 #scalar McNamara 6.22
            D3=common_factor*cot3*F3 #scalar McNamara 6.23
            D4=common_factor*cot4*F4 #scalar McNamara 6.24
            return D1,D2,D3,D4#,phi_i
        
        def spread_factor(Si,Sd):
            #for spherical wave incidence on a straight edge:
            return np.sqrt(Si/(Sd*(Sd+Si)))
        
        # Compute vectors from emitter to reflection point, then to diffraction point to receiver
        vectors, norms = geom.normalize_path(diff_path)
        si = vectors[0, :]
        Si = norms[0] #norm of incident ray
        sd = vectors[1, :]
        Sd = norms[1] #norm of diffracted ray
        
        polygon_1 = rtp.polygons[diff_surfaces_ids[0]]
        polygon_2 = rtp.polygons[diff_surfaces_ids[1]]
        
        diffraction_polygon= polygon_1 #we define this as the 0-face.
        
        n_0=array_utils.normalize(diffraction_polygon.get_normal()) #(vector) unit vector normal to the 0 face
        n_1=array_utils.normalize(polygon_2.get_normal()) #(vector) unit vector normal to the n face
         
        alpha=pi-np.arccos(np.dot(n_0,n_1)) #angles between two planes = pi-angles between their outwards normals
        n=2-alpha/pi #doesn't need to be an integer (McNamara 4.26)
        assert alpha<pi and alpha>0 and n>=1 and n<=2, f"alpha= {alpha*180/pi}, we restrict ourselves to wedges with interior angles in 0,180"
        
        e= vv_normalize(np.diff(corner_points, axis=0).reshape(-1)) #edge
       
        #ensuring that e is such that t_0 points towards the 0 face (McNamara 6.8)
        t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (McNamara 6.8)
        mid_edge=geom.midpoint(corner_points[0], corner_points[1])
        if diffraction_polygon.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8)==False:
            e=-e #flip edge orientation
            t_0=np.cross(n_0,e)       
        assert diffraction_polygon.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8), "edge is wrongly oriented"
        
        #diffraction coefficients components:
        D1,D2,D3,D4=diff_dyadic_components(Si,si,Sd,sd,e,E_i.k,diffraction_polygon,n)
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
        field=(E_i.E.reshape(1,3)@D)*spread_factor(Si,Sd)*np.exp(-1j*E_i.k*Sd) #(vector) McNamara 6.18
        #diffracted field should be less powerful than the incident field
        assert np.linalg.norm(E_i.E)>=np.linalg.norm(field), f"Ed should be less powerful Ei but Ei={np.linalg.norm(E_i.E)} and Ed= {np.linalg.norm(field)} dyadic coeff {D}"
        E_i.E=field.reshape(3,)
        E_i=ElectromagneticField.account_for_trees(E_i,np.array([diff_path[1],diff_path[2]]),rtp.place)
        return E_i

    
    
    
class SolvedField:
    """
    class to store the data about the field computed at the end of a path
    """
    def __init__(self,path=[np.array([0,0,0]),np.array([0,0,0])]):
        self.rx=path[-1]       
        self.tx=path[0]
        self.world_path=path #in world coordinates
        
        self.path_len=geom.path_length(path)
        self.time=self.path_len/c
        
        self.path_type=None 
        self.rx_id=None
        self.field=0
        
        self.antenna_path=None
        
        #elevation and azimuth angles of the TX and RX rays, compared to the antenna axis
        self.rx_antenna=None
        self.tx_antenna=None
        self.tx_el=None
        self.tx_az=None
        self.rx_el=None
        self.rx_az=None
        
        self.power=0
        
    
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
        df.loc[len(df.index)] = [self.rx_id,self.rx,self.path_type,self.time,self.field,self.power] 
        return
    
        
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
    for order in rtp.diffractions:
        for path in range(0,len(rtp.diffractions[receiver][order])):
            the_data=rtp.diffractions[receiver][order][path]  
            the_path=the_data[0]
            path_len=geom.path_length(the_path)
            if path_len<shortest_path_lenght:
                shortest_path_lenght=path_len
                shortest_path=the_path
                path_type="Diff"
                
    for order in rtp.reflections:
        for path in range(0,len(rtp.reflections[receiver][order])):
            the_data=rtp.reflections[receiver][order][path]  
            the_path=the_data[0]
            path_len=geom.path_length(the_path)
            if path_len<shortest_path_lenght:
                shortest_path_lenght=path_len
                shortest_path=the_path
                path_type="Ref"
        
    print(f'the shortest path is a {path_type}')
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
        print('computing LOS')
        #receiver is the index of the receiver NOT its coordinates 
        if rtp.los[receiver]!=[]:     
            sol=SolvedField(rtp.los[receiver][0])
            sol.rx_id=receiver
            sol.path_type="LOS"
            sol.tx_antenna=tx_antenna
            sol.antenna_path=tx_antenna.path_to_antenna_reference_frame(sol.world_path)
            
            E_los=ElectromagneticField.from_path(sol.world_path,tx_antenna)
            E_los=ElectromagneticField.account_for_trees(E_los,sol.world_path,rtp.place)
            sol.field=E_los.E
            
            solved_list.append(sol)  
        return 
       
    #TODO: account for trees for reflections, without duplicate in diffractions  
    #TODO: use correct reference frame
    def compute_reflections(receiver,rtp,solved_list,tx_antenna):
        """
        For each pure reflections path reaching this receiver compute the field
        append the result to solved_list
        """
        print('computing REF')
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
                sol.antenna_path=tx_antenna.path_to_antenna_reference_frame(sol.world_path)
                
                #TODO: which basis to provide the path in?
                E_i=ElectromagneticField.from_path([sol.tx,first_interaction],tx_antenna)
                this_E=ElectromagneticField.my_reflect(E_i,sol.world_path,the_reflection_surfaces_ids,rtp)          
                
                sol.field=this_E.E      
                solved_list.append(sol)                      
        return 
      
        
    #TODO: use correct reference frame
    def compute_diffractions(receiver,rtp,solved_list,tx_antenna):
        """
        For each path containing a diffraction that reaches this receiver
        compute the field and append the result to solved_list
        """
        print('computing DIFF')
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
                    #account for everything (reflections) happening before the diffraction
                    #trees are accounted for directly in my_reflect()
                    E_i=ElectromagneticField.my_reflect(E_0,the_path_before_diff,the_reflection_surfaces_ids,rtp) #E reaching the diffraction point
          
                #account for the diffraction
                this_E=ElectromagneticField.my_diff(E_i,the_diff_path,the_diff_surfaces_ids,the_corner_points,rtp,receiver)
    
                sol=SolvedField(the_path)
                sol.rx_id=receiver
                sol.path_type='R'*(order-1)+'D'
                sol.tx_antenna=tx_antenna
                sol.antenna_path=tx_antenna.path_to_antenna_reference_frame(sol.world_path)
                sol.compute_tx_angles()
                sol.field=this_E.E
                solved_list.append(sol)
        return 
            
   
    #Add trees
    for i in range(0,N_TREES):
        rtp.place.add_tree(tree_size=TREE_SIZE) 
    #compute everything 
    solution=[]
    df=pd.DataFrame(columns=['rx_id','receiver','path_type','time_to_receiver','field_strength','path_power'])
    for receiver in range(0,len(rtp.solved_receivers)):
        print(f"EM SOLVING RECEIVER {receiver}")
        this_solved_list=[] #list containing the fields for all the path between tx and this rx
        tx_antenna=Antenna()
        tx_antenna.position=tx
        rx_antenna=Antenna()
        rx_antenna.position=rtp.receivers[receiver]    
        
        best_path=find_shortest_path(rtp,receiver)
        tx_antenna.align_antenna_to_point(best_path[1])
        rx_antenna.align_antenna_to_point(best_path[-2])
        
        
        compute_LOS(receiver, rtp,this_solved_list,tx_antenna)
        compute_reflections(receiver, rtp,this_solved_list,tx_antenna)
        compute_diffractions(receiver,rtp,this_solved_list,tx_antenna)
        for sol in this_solved_list:
            sol.rx_antenna=rx_antenna
            sol.compute_rx_angles()
            Ae=(lam**2/(4*pi))*RX_GAIN*radiation_pattern(sol.rx_el,sol.rx_az)
            sol.power=(1/2*Z_0)*Ae*(np.linalg.norm(np.real(sol.field)))**2
            #sol.show_antennas_alignement()
            print(f" rx {receiver} path {sol.path_type}: angles theta {sol.rx_el*180/pi} phi {sol.rx_az*180/pi}")
            sol.add_to_df(df)
        solution.append(this_solved_list)
    file_utils.save_df(df,save_name)#saves into the result folder
    return df




def EM_fields_data(df_path):
    df=file_utils.load_df(df_path)
    nreceivers=len(df['rx_id'].unique())
    for receiver in range(nreceivers):
        if receiver in df['rx_id'].values:
            rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]   
            print(f"------------data for receiver {receiver}, with coords {rx_coord}-------------------")
            this_df=df.loc[df['rx_id'] == receiver]     
            
            if(this_df['path_type'].str.contains("LOS").any()):   
                print(f"RX {receiver} is in LOS")
            else:
                print(f"RX {receiver} is in NLOS")
        else:
            print("No path linking TX to RX was found for this receiver")
    return

    
def EM_fields_plots(df_path,order=3,name="notnamed"):
    df=file_utils.load_df(df_path)
    nreceivers=len(df['rx_id'].unique())
    nrows=nreceivers
    ncols=2
    
    fig = plt.figure("EM fields data",figsize=(16,5*nrows))    
    fig.set_dpi(150)
    fig.subplots_adjust(hspace=.5)
    
    i=1
    for receiver in range(0,nreceivers):
        rx_df=df.loc[df['rx_id'] == receiver]
        path_types=rx_df['path_type'].unique()
        ax1 = fig.add_subplot(nrows, ncols,i)
        ax2 = fig.add_subplot(nrows, ncols,i+1)
        i+=2
        width = 0.35
        for path_type in path_types:
            data_for_type=rx_df.loc[rx_df['path_type'] == path_type]
            color_for_type,position,ticks=plot_utils.set_color_for_type(path_type,order)
            #total power from each source
            power=to_db(np.sum(data_for_type["path_power"].values))
            #power=FieldPower.compute_power(data_for_type['field_strength'].values) #receive power in dB    
            ax1.bar(x=position, height=power,width=width,color=color_for_type,label=path_type)     
            #power delay profile
            nelem=len(data_for_type['field_strength'])
            individual_powers=np.zeros(nelem)
            for j in range(nelem):
               individual_powers[j]=to_db(data_for_type['path_power'].values[j]) 
            ax2.stem(data_for_type['time_to_receiver'].values, individual_powers,linefmt=color_for_type,label=path_type,basefmt=" ")
                            
        ax1.set_title(f'Total power from sources RX{receiver}')
        ax1.set_xticks(range(0,len(ticks)), ticks)
        ax1.set_ylabel('Received power [dB]') 
        ax1.grid()
 
        ax2.set_title(f'Power delay Profile RX{receiver}')
        ax2.set_xlabel('time to reach receiver (s)')
        ax2.set_ylabel('Received power [dB]')
        ax2.legend() 
        ax2.grid()
        plt.savefig(f"../results/EM_plots_{name}'.pdf", dpi=300)
        plt.show()
        
    return fig




   
    
    
    
    
    
    
        
        
    
    
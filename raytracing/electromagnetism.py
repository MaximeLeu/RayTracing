from raytracing import array_utils
from raytracing import geometry as geom
import numpy as np
import matplotlib.pyplot as plt
from numpy.dual import norm

import json
import scipy as sc
from scipy.constants import c, mu_0, epsilon_0, pi
from materials_properties import DF_PROPERTIES


#I consider the first medium is always air.
mu_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["mu"].values[0]*mu_0
sigma_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["sigma"].values[0]
epsilon_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["epsilon"].values[0]*epsilon_0


class ElectromagneticField:

    def __init__(self, E=np.array([1, 0, 0], dtype=complex), f=1.8e9, v=c):
        self.E = E
        self.f = f
        self.v = v
        self.k = 2 * pi * f / v
        self.w = 2 * pi * f
        self.lam =v/f #lambda

    @staticmethod
    def from_path(path):
        """
        Given a direct path between two points, computes the electric field at the end of the path using the far field approximation  
        :param path: TX point and point of the first obstruction (or RX if line of sight)
        :type path: numpy.ndarray(shapely.geometry.Points) *shape=(2)
        """
        vector, d = geom.normalize_path(path)
        phi, theta, _ = geom.cartesian_to_spherical(vector * d)

        # Reference axis is z, not x, y plane
        theta -= pi / 2
        if theta < 0:
            theta += 2 * pi
        cphi = np.cos(phi[0])
        sphi = np.sin(phi[0])
        ctheta = np.cos(theta[0])
        stheta = np.sin(theta[0])
        eph = np.array([-sphi, cphi, 0])
        eth = np.array([ctheta * cphi, sphi * ctheta, -stheta])

        EF = ElectromagneticField()
        EF.E = np.exp(-1j*EF.k*d) * (eph + eth) / (4 * pi * d) # TODO missing x(-jk eta) ?
        return EF

    def fresnel_coeffs(E,eta_1,eta_2,theta_i,theta_t,roughness):
        r_par=(eta_2*np.cos(theta_i)-eta_1*np.cos(theta_t)) / (eta_2*np.cos(theta_i)+eta_1*np.cos(theta_t))
        r_per=(eta_2*np.cos(theta_t)-eta_1*np.cos(theta_i)) / (eta_2*np.cos(theta_t)+eta_1*np.cos(theta_i))
  
        rayleigh_criterion= (roughness > E.lam/(8*np.cos(theta_i)))
        if rayleigh_criterion:
            print("applying scattering")
            chi=np.exp(-2*(E.k**2)*(roughness**2)*(np.cos(theta_i))**2)
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
        if place.obstructs_line_path(points):
            print("tree in the way, attenuating")
            attenuation=1 #TODO
            E.E=E.E*attenuation
        return E
    
       
    
    def get_parameters(E,reflection_polygon):    
        mu_2=np.complex(reflection_polygon.properties['mu'])*mu_0
        sigma_2=np.complex(reflection_polygon.properties['sigma'])
        epsilon_2=np.complex(reflection_polygon.properties['epsilon'])*epsilon_0
        roughness=float(reflection_polygon.properties['roughness']) 
        #wave impedance
        eta_1=np.sqrt(1j*E.w*mu_1/(sigma_1+1j*E.w*epsilon_1))
        eta_2=np.sqrt(1j*E.w*mu_2/(sigma_2+1j*E.w*epsilon_2))
        eta=[eta_1, eta_2]
        #propagation constant
        gamma_1=np.sqrt(1j*E.w*mu_1*(sigma_1+1j*E.w*epsilon_1))
        gamma_2=np.sqrt(1j*E.w*mu_2*(sigma_2+1j*E.w*epsilon_2))
        gamma=[gamma_1,gamma_2] 
        
        surface_normal=array_utils.normalize(reflection_polygon.get_normal())
        return mu_2,sigma_2,epsilon_2,eta,gamma,roughness,surface_normal
    
    def my_reflect(E_i,reflections_path,surfaces_ids,rtp):
        """
        Compute the E field at the end of a reflection (reflection can be multiple)
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
            R : TYPE complex number
                DESCRIPTION the dyadic reflection coefficient of the path
           """
            #data:        
            E=ElectromagneticField()
            mu_2,sigma_2,epsilon_2,eta,gamma,roughness,surface_normal=ElectromagneticField.get_parameters(E,reflection_polygon)
            
            #Compute vectors from emitter to reflection point,
            #then to reflection point to receiver
            vectors = np.diff(path, axis=0)
            vectors, norms = geom.normalize_path(path)
            si = vectors[0, :] #normalised incident vector
            sr = vectors[1, :] #normalised reflected vector

            #incident angle and transmission angle
            theta_i=-np.arccos(np.dot(surface_normal, si)) #incident angle
            theta_t=np.arcsin(np.sin(theta_i)*gamma[0]/gamma[1]) #transmission angle from snell law
            
            r_par,r_per=ElectromagneticField.fresnel_coeffs(E,eta[0],eta[1],theta_i,theta_t,roughness)
            
            #dyadic reflection coeff
            e_per=array_utils.normalize(np.cross(-si, sr))
            ei_par= array_utils.normalize(np.cross(e_per,si))
            er_par= array_utils.normalize(np.cross(e_per,-sr))
            
            R=e_per.reshape(3,1)@e_per.reshape(1,3)*r_per - ei_par.reshape(3,1)@er_par.reshape(1,3)*r_par #3x3 matrix
            assert(R.shape==(3,3))
            return R,norms
        
        
        paths=split_reflections_path(reflections_path)
        for i in range(0,len(surfaces_ids)):
            reflection_polygon=rtp.polygons[surfaces_ids[i]]
            R,norms=dyadic_ref_coeff(paths[i],reflection_polygon)
            Si= norms[0]#norm of incident ray
            Sr= norms[1]#norm of reflected ray  
            E_i=ElectromagneticField.account_for_trees(E_i,np.array([paths[i][0],paths[i][1]]),rtp.place) #account for TX-->ref
            field=R@(E_i.E.reshape(3,1))*np.exp(-1j*E_i.k*Sr)*Si/(Si+Sr)
            E_i.E=field.reshape(3,)

        last_path=paths[-1]
        last_path=np.array([last_path[1],last_path[2]])
        E_i=ElectromagneticField.account_for_trees(E_i,last_path,rtp.place) #account for ref-->RX
        return E_i

    def my_diff(E_i,diff_path,diff_surfaces_ids,corner_points,rtp):
        """
        Compute the E field at the end of a diffraction
        """
        def diff_dyadic_components(Si,si,Sd,sd,e,k,phi_i_vv,phi_d_vv,diffraction_polygon):
            """
            Computes the 4 components of the dyadic diffraction coefficient
            e: unit vector tangential to the edge
            si=s' in McNamara= incident ray
            sd=s in McNamara= diffracted ray
            
            """   
            def F(x):
                #efficiently computes the transition function, as demonstrated in
                #https://eertmans.be/research/utd-transition-function/
                #F is defined in Mc 4.72
                factor = np.sqrt(np.pi / 2)
                sqrtx = np.sqrt(x)
                S, C = sc.special.fresnel(sqrtx / factor)
                transition_function=2j * sqrtx* np.exp(1j * x)* (factor * ((1 - 1j) / 2 - C + 1j * S))
                return transition_function
            
            def A(x,sign):
            #definition is given in MC 4.66
                match sign:
                    case "minus":
                        N=round((-pi+x)/(2*n*pi)) #TODO n
                    case "plus":
                        N=round((pi+x)/(2*n*pi))
                    case _:
                        raise ValueError('sign should either be plus or minus')                
                result=2*(np.cos(n*pi*N-x/2))**2
                return result
            
            def cot(x):
                return 1/np.tan(x)
            
            #beta_0 and beta_0' have the same norm but are not the same vectors.
            beta_0=np.arccos(np.dot(si,e)) #(scalar) angle between the incident ray and the edge (Mc 4.19)
            
            assert(beta_0<=pi and beta_0>=0)
            n_0=array_utils.normalize(diffraction_polygon.get_normal()) #(vector) unit vector normal to the 0-face
            t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (Mc 6.8)
            
            
            #unit vector components of si and sd lying in the plane perpendicular to the edge (=s_t' and s_t in McNamara)
            si_t=si-np.dot(si,e)*e 
            si_t=si_t/np.dual.norm(si_t) #(vector) Mc 6.9
            sd_t=sd-np.dot(sd,e)*e
            sd_t=sd_t/np.dual.norm(sd_t) #(vector) Mc 6.10
            
            phi_i=pi-(pi-np.arccos(np.dot(-si_t,t_0)))*np.sign(np.dot(-si_t,n_0)) #(scalar) Mc 6.11 incident angle with respect to the O-face
            phi_d=pi-(pi-np.arccos(np.dot(sd_t,t_0)))*np.sign(np.dot(sd_t,n_0)) #(scalar) Mc 6.12 diffraction angle with respect to the O-face
            phi_i=phi_i%(2*pi)#get back in the 0-2pi range
            phi_d=phi_d%(2*pi)
           
            
            #TODO phi_d and phi_i are somehow the same??? see text after 4.39
            alpha=pi/2 #TODO (radians) interior angle of the wedge             
            n=2-alpha/pi #doesn't need to be an integer (Mc 4.26)
            assert(alpha<=pi) #we restrict ourselves to wedges with interior angles <=180
            
            # text right after Mc 6.12 for these asserts:
                #TODO
            print(f"si_t {si_t}, sd_t {sd_t}")
            print(f"t_0 {t_0}, n_0 {n_0}")
            print(f"dot {np.arccos(np.dot(-si_t,t_0))} dot2 {np.arccos(np.dot(sd_t,t_0))}")
            print(f"phi_i {phi_i} phi_d {phi_d}")
            print(f"n {n*pi}")
            #assert(phi_i>=0 and phi_i<=n*pi)  #TODO
            #assert(phi_d>=0 and phi_d<=n*pi) #TODO

            
            #components of the diffraction coefficients:
            common_factor=-np.exp(-1j*pi/4)/(2*n*np.sqrt(2*pi*k)*np.sin(beta_0)) #scalar
            L=((np.sin(beta_0))**2)*Si*Sd/(Si+Sd) #Scalar Mc 6.26, I neglect the curvature of the edge.
        
            cot1=cot((pi+(phi_d-phi_i))*1/(2*n)) #scalar
            cot2=cot((pi-(phi_d-phi_i))*1/(2*n))
            cot3=cot((pi+(phi_d+phi_i))*1/(2*n))
            cot4=cot((pi-(phi_d+phi_i))*1/(2*n))
            
            F1=F(k*L*A(phi_d-phi_i,"plus")) #scalar
            F2=F(k*L*A(phi_d-phi_i,"minus"))
            F3=F(k*L*A(phi_d+phi_i,"plus"))
            F4=F(k*L*A(phi_d+phi_i,"minus"))
            
            D1=common_factor*cot1*F1 #scalar Mc 6.21
            D2=common_factor*cot2*F2 #scalar Mc 6.22
            D3=common_factor*cot3*F3 #scalar Mc 6.23
            D4=common_factor*cot4*F4 #scalar Mc 6.24
            return D1,D2,D3,D4
        
        
        # Compute vectors from emitter to reflection point, then to diffraction point to receiver
        vectors, norms = geom.normalize_path(diff_path)
        si = vectors[0, :]
        Si = norms[0] #norm of incident ray
        sd = vectors[1, :]
        Sd = norms[1] #norm of diffracted ray
        
        polygon_1 = rtp.polygons[diff_surfaces_ids[0]]
        polygon_2 = rtp.polygons[diff_surfaces_ids[1]]
        edge= np.diff(corner_points, axis=0).reshape(-1)
        
        e = array_utils.normalize(edge) #TODO: vector tangential to the edge, is it correct?
        
        #parameters
        diffraction_polygon= polygon_1#TODO, what if they don't have the same properties?
        mu_2,sigma_2,epsilon_2,eta,gamma,roughness,surface_normal=ElectromagneticField.get_parameters(E_i,diffraction_polygon)
        
        theta_i=1 #TODO use the correct angles
        theta_t=1
        r_par,r_per=ElectromagneticField.fresnel_coeffs(E_i,eta[0],eta[1],theta_i,theta_t,roughness)
        
        #dyadic diffraction coefficient:
        phi_i_vv=np.cross(-e,si)
        phi_i_vv=phi_i_vv/np.dual.norm(phi_i_vv) #(vector) Mc 6.2
        phi_d_vv=np.cross(e,sd)
        phi_d_vv=phi_d_vv/np.dual.norm(phi_d_vv) #(vector) Mc 6.4
        
        D1,D2,D3,D4=diff_dyadic_components(Si,si,Sd,sd,e,E_i.k,phi_i_vv,phi_d_vv,diffraction_polygon)
        
        beta_0_i_vv=np.cross(phi_i_vv,si) # (vector) Mc 6.3
        beta_0_d_vv=np.cross(phi_d_vv,sd) #(vector) Mc 6.5
        
        D_per=D1+D2+r_per*(D3+D4) #(scalar) Mc 6.20
        D_par=D1+D2+r_par*(D3+D4) #(scalar) Mc 6.20
        
        D=beta_0_i_vv.reshape(3,1)@beta_0_d_vv.reshape(1,3)*(-D_per) - phi_i_vv.reshape(3,1)@phi_d_vv.reshape(1,3)*D_par # (3x3 matrix) Mc 6.19
        assert(D.shape==(3,3))
        #field reaching receiver after a diffraction:
        field=(E_i.E.reshape(1,3)@D)*np.sqrt(Si/(Sd*(Sd+Si)))*np.exp(-1j*E_i.k*Sd) #(vector) Mc 6.18
        E_i.E=field.reshape(3,)
        E_i=ElectromagneticField.account_for_trees(E_i,np.array([diff_path[1],diff_path[2]]),rtp.place)
        return E_i





def my_field_computation(rtp):
    #Computes the resulting field at the receiver from all reflections and diffractions
    reflections = rtp.reflections
    diffractions = rtp.diffractions
    fields=[]
    
    #Add trees
    trees=10
    for i in range(0,trees):
        rtp.place.add_tree(tree_size=2)    

    fig = plt.figure("place with trees")
    fig.set_dpi(300)
    ax = fig.add_subplot(projection='3d')
    rtp.place.center_3d_plot(ax)
    ax = rtp.place.plot3d(ax=ax)    

    
    # computes field for each reflection and diffraction
    for receiver in range(0,len(rtp.place.set_of_points)):
        #compute field resulting from all reflections:         
        reflections_field=ElectromagneticField()
        reflections_field.E=0    
        for order in reflections[receiver]:
            for path in range(0,len(reflections[receiver][order])):
                the_data=reflections[receiver][order][path]
                the_path=the_data[0]
                the_reflection_surfaces_ids=the_data[1]
                tx =the_path[0] #first point of the reflection
                first_interaction=the_path[1] #second point of the path
                assert(np.array_equal(rtp.emitter[0],tx))
                E_i=ElectromagneticField.from_path([tx,first_interaction])
                this_E=ElectromagneticField.my_reflect(E_i,the_path,the_reflection_surfaces_ids,rtp)
                reflections_field.E+=this_E.E
                
                
        #compute all diffractions
        diffractions_field=ElectromagneticField()
        diffractions_field.E=0
        for order in range(0,len(diffractions[receiver])):     
            for path in range(0,len(diffractions[receiver][order])):
                the_data=diffractions[receiver][order][path]
                the_path=the_data[0]
                the_reflection_surfaces_ids=the_data[1]
                the_diff_surfaces_ids=the_data[2]
                the_corner_points=the_data[3]
                
                #assumes diff always happen last
                the_path_before_diff=the_path[0:-1] 
                the_diff_path=the_path[-3:]
                
                tx =the_path[0] #first point of the reflection
                first_interaction=the_path[1] #second point of the path
                
                E_0=ElectromagneticField.from_path([tx,first_interaction])
                assert len(the_path>=3), "path doesn't have the right shape"
                if(len(the_path==3)):
                    E_i=ElectromagneticField.account_for_trees(E_0,np.array([the_path[0],the_path[1]]),rtp.place)
                else:
                    #trees are accounted for directly in my_reflect()
                    E_i=ElectromagneticField.my_reflect(E_0,the_path_before_diff,the_reflection_surfaces_ids,rtp) #E reaching the diffraction point
          
                this_E=ElectromagneticField.my_diff(E_i,the_diff_path,the_diff_surfaces_ids,the_corner_points,rtp)
                diffractions_field.E+=this_E.E
                
        
        total_field=ElectromagneticField()
        total_field.E=diffractions_field.E+reflections_field.E
        fields.append(total_field)
        E_los=ElectromagneticField.from_path(rtp.los[receiver][0])
        
        print("-----------------------------------------------------------")
        print(f"------------data for receiver {receiver}-------------------")
        print("-----------------------------------------------------------")
        print(f"Field if there was a straigth line of sight {E_los.E}")
        print(f"field resulting from all reflections is {reflections_field.E}")
        print(f"field resulting from all diffractions is {diffractions_field.E}") 
        print(f"Total field is {total_field.E}") 
        
        print("")
        print("amplitude of fields:")
        print(f"from reflections {np.dual.norm(reflections_field.E)}")
        print(f"from diffractions {np.dual.norm(diffractions_field.E)}")
        print(f"total field {np.dual.norm(total_field.E)}")     
        print(f"from LOS {np.dual.norm(E_los.E)}")
                     
        
    return fields



    
    
    
    
    
    
    
    
    
    
    
        
        
    
    
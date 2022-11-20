#self written imports
from raytracing import array_utils
from raytracing import geometry as geom
from materials_properties import DF_PROPERTIES, FREQUENCY,N_TREES,TREE_SIZE

#packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import scipy as sc
from scipy.constants import c, mu_0, epsilon_0, pi
pd.set_option('display.max_columns', 10)
pd.options.display.float_format = '{:.2g}'.format


#I consider the first medium is always air.
mu_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["mu"].values[0]*mu_0
sigma_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["sigma"].values[0]
epsilon_1=DF_PROPERTIES.loc[DF_PROPERTIES["material"]=="air"]["epsilon"].values[0]*epsilon_0

DB_REFERENCE=1 #initially radiated field strenght (V/m)

class ElectromagneticField:

    def __init__(self, E=np.array([1, 0, 0], dtype=complex), f=FREQUENCY, v=c):
        self.E = E
        self.f = f
        self.v = v
        self.k = 2 * pi * f / v
        self.w = 2 * pi * f
        self.lam =v/f #lambda

    @staticmethod
    def from_path(path):
        """
        Given a direct path between TX and RX, computes the electric field at RX using the far field approximation  
        :param path: TX point and point of the first obstruction (or RX if line of sight)
        :type path: numpy.ndarray(shapely.geometry.Points) *shape=(2)
        """
        vector, d = geom.normalize_path(path)
        
        # phi, theta, r = geom.cartesian_to_spherical(vector * d) #TODO, ca n'as pas de sens de convertir un vecteur en coordonnées sphériques?? Ca donne simplement sa norme..
        # # Reference axis is z, not x, y plane
        # theta -= pi / 2
        # if theta < 0:
        #     theta += 2 * pi
        # cphi = np.cos(phi[0])
        # sphi = np.sin(phi[0])
        # ctheta = np.cos(theta[0])
        # stheta = np.sin(theta[0])
        
        # #spherical unit vectors
        # eph = np.array([-sphi, cphi, 0])
        # eth = np.array([ctheta * cphi, sphi * ctheta, -stheta])
        # eta=np.sqrt(mu_1/(epsilon_1)) #376.730 = free space impedance
        # assert eta>376 and eta<377, "strange free space impedance value"
        # #EF.E =1j*EF.k*eta*np.exp(-1j*EF.k*d) * (eph + eth) / (4 * pi * d)
          
        
        E_0 = ElectromagneticField()
        E_0.E=np.array([DB_REFERENCE, 0, 0]) #intially radiated field strenght (V/m)
        EF=ElectromagneticField()
        EF.E=E_0.E*np.exp(-1j*E_0.k*d)/d
        return EF

    def fresnel_coeffs(E,theta_i,roughness,epsilon_eff_2):
        assert theta_i>0 and theta_i<pi/2, f"theta_i={theta_i*180/pi} degrees, but should between 0,90"
        #assuming the first medium is air
        sqrt=np.sqrt(epsilon_eff_2-(np.sin(theta_i))**2)
        cosi=np.cos(theta_i)
        r_par=(epsilon_eff_2*cosi-sqrt) / (epsilon_eff_2*cosi+sqrt)    #Rh 
        r_per=(cosi-sqrt) / (cosi+sqrt)    #Rs 
        r_par=np.linalg.norm(r_par)
        r_per=np.linalg.norm(r_per)
        print(f"rpar {r_par} r_per {r_per}")
        assert r_par<=1 and r_per<=1, f"fresnel coeffs are greater than 1: r_par {r_par} r_per {r_per}"
        
        rayleigh_criterion= (roughness > E.lam/(8*cosi))
        if rayleigh_criterion:
            print("applying scattering")
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
                #attenuation=1/(10**(L_db/10)) #TODO   
                attenuation=1
                print(f"attenuation: {L_db} dB, thus attenuation factor= {attenuation} ")
                E.E=E.E*attenuation
            
        return E
    
       
    
    def get_parameters(E,reflection_polygon):    
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
            R : TYPE complex number
                DESCRIPTION the dyadic reflection coefficient of the path
           """
            #data:        
            E=ElectromagneticField()
            mu_2,sigma_2,epsilon_2,epsilon_eff_2,roughness,surface_normal=ElectromagneticField.get_parameters(E,reflection_polygon)
            #Compute vectors from emitter to reflection point,
            #then from reflection point to receiver
            vectors = np.diff(path, axis=0)
            vectors, norms = geom.normalize_path(path)
            si = vectors[0, :] #normalised incident vector
            sr = vectors[1, :] #normalised reflected vector

            #incident angle and transmission angle
            theta_i=np.arccos(np.dot(surface_normal, -si)) #incident angle
            r_par,r_per=ElectromagneticField.fresnel_coeffs(E,theta_i,roughness,epsilon_eff_2)
            
            #dyadic reflection coeff
            e_per=array_utils.normalize(np.cross(si, sr))
            ei_par= array_utils.normalize(np.cross(e_per,si))
            er_par= array_utils.normalize(np.cross(e_per,-sr))
            
            R=e_per.reshape(3,1)@e_per.reshape(1,3)*r_per - ei_par.reshape(3,1)@er_par.reshape(1,3)*r_par #3x3 matrix
            assert(R.shape==(3,3))
            return R,norms, e_per,ei_par, er_par,r_par,r_per
        
        E_r=ElectromagneticField()
        E_r.E=E_i.E
        paths=split_reflections_path(reflections_path)
        for i in range(0,len(surfaces_ids)):
            reflection_polygon=rtp.polygons[surfaces_ids[i]]
            R,norms,_,_,_,_,_=dyadic_ref_coeff(paths[i],reflection_polygon)
            Si= norms[0]#norm of incident ray
            Sr= norms[1]#norm of reflected ray  
            E_r=ElectromagneticField.account_for_trees(E_r,np.array([paths[i][0],paths[i][1]]),rtp.place) #account for TX-->ref # incident field at the point of reflection.
            field=(E_r.E.reshape(1,3)@R)*np.exp(-1j*E_r.k*Sr)*Si/(Si+Sr)
            #reflected field should be of lesser intensity than the incident field
            assert np.linalg.norm(E_i.E)>=np.linalg.norm(field), f"incident field intensity {np.linalg.norm(E_i.E)} reflected field {np.linalg.norm(field)} dyadic coeff {R}"
            E_r.E=field.reshape(3,)          
        last_path=paths[-1] #[Tx,R,RX]
        last_path=np.array([last_path[1],last_path[2]]) #[R,RX]
        E_r=ElectromagneticField.account_for_trees(E_r,last_path,rtp.place) #account for ref-->RX
        #print(f"REAL incident field intensity {np.linalg.norm(E_i.E.real)} reflected field {np.linalg.norm(field.real)}")
        #print(f"IM incident field intensity {np.linalg.norm(E_i.E.imag)} reflected field {np.linalg.norm(field.imag)}")
        
        #TODO: should I take the norm of the real part? or of both?
        
        # #ttttttttttttesssst
        # def explicit_compute(E_i):
        #     paths=split_reflections_path(reflections_path)
        #     E_test=ElectromagneticField()
        #     E_test.E=E_i.E
        #     for i in range(0,len(surfaces_ids)):
        #         reflection_polygon=rtp.polygons[surfaces_ids[i]]
        #         R,norms, e_per,ei_par, er_par,r_par,r_per =dyadic_ref_coeff(paths[i],reflection_polygon)
        #         Si= norms[0]#norm of incident ray
        #         Sr= norms[1]#norm of reflected ray  
        #         E_test=ElectromagneticField.account_for_trees(E_test,np.array([paths[i][0],paths[i][1]]),rtp.place) #account for TX-->ref
                
        #         #other computation
        #         cst=np.exp(-1j*E_i.k*Sr)*Si/(Si+Sr)
        #         Er_par=np.dot(E_test.E,er_par)*cst*r_par
        #         Er_per=np.dot(E_test.E,e_per)*cst*r_per
        #         E_test.E=Er_par*er_par+Er_per*e_per
        #         E_test.E=field.reshape(3,)   
        #         assert np.linalg.norm(E_i.E)>=np.linalg.norm(E_test.E), f"incident field intensity {np.linalg.norm(E_i.E)} reflected field {np.linalg.norm(E_test.E)}"
                
        #     last_path=paths[-1]
        #     last_path=np.array([last_path[1],last_path[2]])
        #     E_test=ElectromagneticField.account_for_trees(E_test,last_path,rtp.place) #account for ref-->RX
            
        #     return E_test
        
        # E_test=explicit_compute(E_i)
        # print(f"E using dyadic {np.linalg.norm(E_i.E)} and E using my compute {np.linalg.norm(E_test.E)}")
        
        return E_r






    def my_diff(E_i,diff_path,diff_surfaces_ids,corner_points,rtp):
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
            assert(beta_0<=pi and beta_0>=0)
            
            n_0=array_utils.normalize(diffraction_polygon.get_normal()) #(vector) unit vector normal to the 0-face
            t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (McNamara 6.8)

            #unit vector components of si and sd lying in the plane perpendicular to the edge (=s_t' and s_t in McNamara)
            si_t=si-np.dot(si,e)*e 
            si_t=si_t/np.linalg.norm(si_t) #(vector) McNamara 6.9
            sd_t=sd-np.dot(sd,e)*e
            sd_t=sd_t/np.linalg.norm(sd_t) #(vector) McNamara 6.10
            
            #TODO: ocasionnal problems with those angles, IDK why
            phi_i=pi-(pi-np.arccos(np.dot(-si_t,t_0)))*np.sign(np.dot(-si_t,n_0)) #(scalar) McNamara 6.11 incident angle with respect to the O-face 
            phi_d=pi-(pi-np.arccos(np.dot(sd_t,t_0)))*np.sign(np.dot(sd_t,n_0)) #(scalar) McNamara 6.12 diffraction angle with respect to the O-face
            phi_i=phi_i%(2*pi)#get back in the 0-2pi range
            phi_d=phi_d%(2*pi)

            # text right after Mc 6.12 for these asserts:
            assert phi_i>=0 and phi_i<=n*pi , f"phi_i ={phi_i*180/pi} cannot be inside the wedge, alpha={(2-n)*pi*180/pi}"
            assert phi_d>=0 and phi_d<=n*pi, f"phi_d {phi_d*180/pi} cannot be inside the wedge, alpha={(2-n)*pi*180/pi}" 
            
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
            return D1,D2,D3,D4,phi_i
        
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
         
        alpha=np.arccos(np.dot(n_0,n_1)) #angles between two planes = angles between their normals
        n=2-alpha/pi #doesn't need to be an integer (McNamara 4.26)
        assert alpha<=pi and n>=1 and n<=2, f"alpha= {alpha*180/pi}, we restrict ourselves to wedges with interior angles <=180"
        
        edge= np.diff(corner_points, axis=0).reshape(-1)
        e = edge/np.linalg.norm(edge)
       
        #ensuring that e is such that t_0 points towards the 0 face (McNamara 6.8)
        t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (McNamara 6.8)
        mid_edge=geom.midpoint(corner_points[0], corner_points[1])
        if diffraction_polygon.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8)==False:
            e=-e #flip edge orientation
            t_0=np.cross(n_0,e)       
        assert diffraction_polygon.contains_point(mid_edge+t_0*(1e-6), check_in_plane=True, plane_tol=1e-8), "edge is wrongly oriented"
        
        #diffraction coefficients components:
        D1,D2,D3,D4,phi_i=diff_dyadic_components(Si,si,Sd,sd,e,E_i.k,diffraction_polygon,n)
        
        #TODO: consider the edge as a other than a PEC
        r_per=1
        r_par=1
        #diffraction coefficients  
        D_per=D1+D2+r_per*(D3+D4) #(scalar) McNamara 6.20
        D_par=D1+D2+r_par*(D3+D4) #(scalar) McNamara 6.20
        
        #Edge fixed coordinate system
        phi_i_vv=np.cross(-e,si)
        phi_i_vv=phi_i_vv/np.linalg.norm(phi_i_vv) #(vector) McNamara 6.2
        phi_d_vv=np.cross(e,sd)
        phi_d_vv=phi_d_vv/np.linalg.norm(phi_d_vv) #(vector) McNamara 6.4 
        beta_0_i_vv=np.cross(phi_i_vv,si) 
        beta_0_i_vv=beta_0_i_vv/np.linalg.norm(phi_i_vv)# (vector) McNamara 6.3
        beta_0_d_vv=np.cross(phi_d_vv,sd) 
        beta_0_d_vv=beta_0_d_vv/np.linalg.norm(phi_d_vv)#(vector) McNamara 6.5
        
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

    
    
def my_field_computation(rtp,save_name=None):
    #Computes the resulting field at the receiver from all reflections and diffractions in V/m
    reflections = rtp.reflections
    diffractions = rtp.diffractions
    
    df_pdp = pd.DataFrame(columns=['total_len','time_to_receiver','path_power','path_type','receiver','rx_id'])  #dataframe storing all results
    
    #Add trees
    for i in range(0,N_TREES):
        rtp.place.add_tree(tree_size=TREE_SIZE)    
    
    
    
    # computes field for each reflection and diffraction
    #for receiver in range(0,len(rtp.place.set_of_points)):
    for receiver in range(0,len(rtp.solved_receivers)):
        exist_path=False
        rx=rtp.solved_receivers[receiver]
        pretty_rx=np.array2string(rx,separator=',')
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
                
                path_length=geom.path_length(the_path)             
                df_pdp.loc[len(df_pdp.index)] = [path_length,path_length/this_E.v,np.linalg.norm(this_E.E),'R'*order,pretty_rx,(receiver)] #TODO: should I take the norm?
                exist_path=True
                
        #compute all diffractions
        diffractions_field=ElectromagneticField()
        diffractions_field.E=0     
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
                
                tx =the_path[0] #first point of the reflection
                first_interaction=the_path[1] #second point of the path
                
                E_0=ElectromagneticField.from_path([tx,first_interaction])
                assert len(the_path)>=3, "path doesn't have the right shape"
                if len(the_path)==3:
                    #TX-diff-RX
                    E_i=ElectromagneticField.account_for_trees(E_0,np.array([the_path[0],the_path[1]]),rtp.place)
                else:
                    #account for everything (reflections) happening before the diffraction
                    #trees are accounted for directly in my_reflect()
                    E_i=ElectromagneticField.my_reflect(E_0,the_path_before_diff,the_reflection_surfaces_ids,rtp) #E reaching the diffraction point
          
                #account for the diffraction
                this_E=ElectromagneticField.my_diff(E_i,the_diff_path,the_diff_surfaces_ids,the_corner_points,rtp)
                diffractions_field.E+=this_E.E
                
                path_length=geom.path_length(the_path)
                df_pdp.loc[len(df_pdp.index)] = [path_length,path_length/this_E.v ,np.linalg.norm(this_E.E),'R'*(order-1)+'D',pretty_rx,receiver]
                exist_path=True
        
        total_field=ElectromagneticField()
        total_field.E=diffractions_field.E+reflections_field.E
        
        #add LOS to total field
        LOS=(rtp.los[receiver]!=[])  
       
        if LOS==True:
            print("there is line of sight")
            E_los=ElectromagneticField.from_path(rtp.los[receiver][0])
            E_los=ElectromagneticField.account_for_trees(E_los,rtp.los[receiver][0],rtp.place)       
            path_length=geom.path_length(rtp.los[receiver][0])       
            print(f"fields Rx {rx}")
            df_pdp.loc[len(df_pdp.index)] = [path_length,path_length/this_E.v, np.linalg.norm(E_los.E),'LOS',pretty_rx,receiver]
            exist_path=True
            total_field.E+=E_los.E
        else:
            print("there is NO line of sight")
            E_los=ElectromagneticField()
            E_los.E=0
         
        if exist_path==False:
            #if there was no path at all (no los, no reflection, no diffraction) between the RX and the TX
            df_pdp.loc[len(df_pdp.index)] = [1,1,0,'No_path_possible',pretty_rx,receiver]
            
            
    if save_name is not None:
        df_pdp.to_csv(f"../results/{save_name}",index=False)
    return df_pdp



def vm_to_db(vm):
    """
    converts V/m to dB (normalised to DB_reference)
    Usually DB_reference is the TX power, which is often chosen as 1. 
    """
    #avoid computing log(0)
    vm1=np.asarray([vm]) if np.isscalar(vm) else np.asarray(vm)
    ind=np.where(vm1!=0)
    db=np.ones(len(vm1))*np.NINF
    for i in ind:
        db[i]=20*np.log10(vm1[i]/DB_REFERENCE)
        
    return db[0] if np.isscalar(vm) else db
   
    
    


def db_to_vm(db):
    """
    converts the unit dB micro V/m to V/m
    """
    vm=DB_REFERENCE*10**(db/20)
    return vm
    
def EM_fields_data(df):
    #df = pd.DataFrame(columns=['total_len','time_to_receiver','path_power','path_type','receiver'])
    nreceivers=len(df['rx_id'].unique())
    for receiver in range(0,nreceivers):
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        rx_coord=rx_coord[1:-1] #remove brackets
        rx_coord=np.fromstring(rx_coord, sep=',')
        print(f"------------data for receiver {receiver}, with coords {rx_coord}-------------------")
        
        
        this_df=df.loc[df['rx_id'] == receiver]
        
        if(this_df['path_type'].str.contains("No_path_possible").any()):
            print("No path linking TX to RX was found for this receiver")
            continue
        
        if(this_df['path_type'].str.contains("LOS").any()):
            los=this_df.loc[this_df['path_type'] == "LOS"]["path_power"].values[0]
            print("there is line of sight")
           
        else:
            print('NO LINE OF SIGHT')
            los=0
        
        
        total=np.sum(this_df["path_power"])
        pure_reflections = np.sum(this_df[this_df['path_type'].str.contains('D|LOS',regex=True)==False]["path_power"])
        pure_diffractions = np.sum(this_df[this_df['path_type'].str.contains('R|LOS',regex=True)==False]["path_power"])
        mixed = np.sum(this_df[this_df['path_type'].str.contains('RD')]["path_power"])
        
        print(this_df)
        print(f"total field {total} V/m")
        print(f"from LOS {los} V/m")
        print(f"from PURE reflections {pure_reflections} V/m")
        print(f"from PURE diffractions {pure_diffractions} V/m")
        print(f"from paths with reflection(s) and diffraction {mixed} V/m")
        print()
        print(f"total field {vm_to_db(total)} dB")
        print(f"from LOS {vm_to_db(los)} dB")
        print(f"from PURE reflections {vm_to_db(pure_reflections)} dB")
        print(f"from PURE diffractions {vm_to_db(pure_diffractions)} dB")
        print(f"from paths with reflection(s) and diffraction {vm_to_db(mixed)} dB")
    
    
#TODO
def EM_fields_plots(df):
    colors=list(mcolors.TABLEAU_COLORS) #10 different colors
    
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
        assert len(path_types)<=len(colors), "too many path types, not enough colors for plotting power sources"
        
        ax1 = fig.add_subplot(nrows, ncols,i)
        ax2 = fig.add_subplot(nrows, ncols,i+1)
        
        
        i+=2
        count=0
        width = 0.35
        for path_type in path_types:
            data_for_type=rx_df.loc[rx_df['path_type'] == path_type]
            total_power=np.sum(data_for_type['path_power'].values)
            total_power=vm_to_db(total_power)
            color_for_type=colors[count]
            count=count+1
            #total power from each source
            ax1.bar(x=count, height=total_power,width=width,color=color_for_type,label=path_type)
            #power delay profile
            ax2.stem(data_for_type['time_to_receiver'].values, vm_to_db(data_for_type['path_power'].values),color_for_type,label=path_type,basefmt=" ")
            
            
        
         
        x=np.arange(1,len(path_types)+1, 1)   
        labels = path_types
        ax1.set_xticks(x, labels)
        ax1.set_title(f'Total power from sources RX{receiver}')
        ax1.set_ylabel('Received power [dB]') 
        ax1.grid()
        #ax1.set_ylim(0,20)
        
        ax2.set_title(f'Power delay Profile RX{receiver}')
        ax2.set_xlabel('time to reach receiver (s)')
        ax2.set_ylabel('Received power [dB]')
        ax2.legend() 
        ax2.grid()
        #ax2.set_ylim(0,20) 
        plt.savefig("../results/EM_plots'.pdf", dpi=300)
        plt.show()
    return fig




   
    
    
    
    
    
    
        
        
    
    
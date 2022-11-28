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

TX_POWER=1 #initially radiated field strenght in W
TX_GAIN=4*pi*1/((pi/6)**2) #4pi/(az*el), where az and el are the 3db beamdidths angles in radians 
RX_GAIN=4*pi*1/((pi/9)**2) #20 degree beamwidth 
#EIRP=TX_POWER*TX_GAIN


class ElectromagneticField:

    def __init__(self, E=np.array([1, 0, 0], dtype=complex), f=FREQUENCY, v=c):
        self.E = E
        self.f = f
        self.v = v
        self.k = 2 * pi * f / v
        self.w = 2 * pi * f
        self.lam =v/f 

    @staticmethod
    def from_path(path):
        """
        Given a direct path between TX and RX, computes the electric field at RX using the far field approximation  
        :param path: TX point and point of the first obstruction (or RX if line of sight)
        :type path: numpy.ndarray(shapely.geometry.Points) *shape=(2)
        """
        #TODO: account for directivity here.
        vector, d = geom.normalize_path(path)
        #assuming isotropic radiator:
        Z_0=120*pi
        init=np.sqrt(2*Z_0*TX_POWER*TX_GAIN/(4*pi*1)) #    
        E_0 = ElectromagneticField()
        E_0.E=np.array([init, 0, 0]) #initially radiated field strenght (V/m) (peak to peak)
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
            R : TYPE complex number
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
            theta_i=np.arccos(np.dot(surface_normal, -si)) #incident angle
            r_par,r_per=ElectromagneticField.fresnel_coeffs(E_i,theta_i,roughness,epsilon_eff_2)
            
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
            assert(beta_0<=pi and beta_0>=0)
            
            n_0=array_utils.normalize(diffraction_polygon.get_normal()) #(vector) unit vector normal to the 0-face
            t_0=np.cross(n_0,e) #(vector) unit vector tangential to the O-face (McNamara 6.8)

            #unit vector components of si and sd lying in the plane perpendicular to the edge (=s_t' and s_t in McNamara)
            si_t=si-np.dot(si,e)*e 
            si_t=si_t/np.linalg.norm(si_t) #(vector) McNamara 6.9
            sd_t=sd-np.dot(sd,e)*e
            sd_t=sd_t/np.linalg.norm(sd_t) #(vector) McNamara 6.10
            
            
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
                return 0,0,0,0
                #raise
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
        #D1,D2,D3,D4,phi_i=diff_dyadic_components(Si,si,Sd,sd,e,E_i.k,diffraction_polygon,n)
        D1,D2,D3,D4=diff_dyadic_components(Si,si,Sd,sd,e,E_i.k,diffraction_polygon,n)
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

    
    
    
class solved_field:
    def __init__(self,path):
        self.rx=path[-1]       
        self.tx=path[0]
        self.path=path
        self.path_len=geom.path_length(path)
        self.time=self.path_len/c
        
        self.path_type=None 
        self.rx_id=None
        self.field=None
        
        self.tx_el=None
        self.tx_az=None
        self.rx_el=None
        self.rx_az=None
     
    def compute_angles(self,RX_basis,TX_basis,tx):
        second_point=self.path[1]
        penultimate_point=self.path[-2] 
        self.rx_el,self.rx_az=solved_field.antenna_incident_angles(RX_basis,penultimate_point-self.rx)
        self.tx_el,self.tx_az=solved_field.antenna_incident_angles(TX_basis,second_point-tx)
        return
    
    
    def my_field_computation(rtp,save_name="problem_fields"):
        #Computes the resulting field at the receiver from all reflections and diffractions in V/m 
       
        #Add trees
        for i in range(0,N_TREES):
            rtp.place.add_tree(tree_size=TREE_SIZE)    
            
        def compute_LOS(receiver,rtp,the_df,solved_list):
            #receiver is the index of the receiver NOT its coordinates 
            rx=rtp.solved_receivers[receiver]
            best = {'field': 0, 'path': None}
            if rtp.los[receiver]!=[]:
                print(f"RX {rx} is in LOS")
                # tx=rtp.los[receiver][0][0]
                # E_los=ElectromagneticField.from_path([tx,rx])
                # E_los=ElectromagneticField.account_for_trees(E_los,rtp.los[receiver][0],rtp.place)       
                # path_length=geom.path_length(rtp.los[receiver][0]) 
                # the_df.loc[len(the_df.index)] = [path_length,path_length/E_los.v,np.array(E_los.E,dtype=complex),'LOS',rx,receiver,rx,tx,0,0,0,0] 
                     
                sol=solved_field(rtp.los[receiver][0])
                sol.rx_id=receiver
                sol.path_type="LOS"
                
                E_los=ElectromagneticField.from_path([tx,rx])
                E_los=ElectromagneticField.account_for_trees(E_los,rtp.los[receiver][0],rtp.place)
                sol.field=E_los.E
                solved_list.append(sol)
                
                best['field']=np.linalg.norm(E_los.E)
                best['path']=[tx,rx]
                
                return best
            print(f"RX {rx} is in NLOS")
            return best
           
        #TODO: account for trees for reflections, without duplicate in diffractions  
        def compute_reflections(receiver,rtp,the_df,solved_list):
            #rx=rtp.solved_receivers[receiver]
            reflections = rtp.reflections
            best = {'field': 0, 'path': None}
            for order in reflections[receiver]:
                for path in range(0,len(reflections[receiver][order])):
                    the_data=reflections[receiver][order][path]
                    the_path=the_data[0]
                    the_reflection_surfaces_ids=the_data[1]
                    tx =the_path[0]
                    first_interaction=the_path[1] #second point of the path
                    E_i=ElectromagneticField.from_path([tx,first_interaction])
                    this_E=ElectromagneticField.my_reflect(E_i,the_path,the_reflection_surfaces_ids,rtp)          
                    
                    
                    # path_length=geom.path_length(the_path)   
                    # the_df.loc[len(the_df.index)] = [path_length,path_length/this_E.v,np.array(this_E.E,dtype=complex),'R'*order,rx,receiver,the_path[1],the_path[-2],0,0,0,0]
                    
                    sol=solved_field(the_path)
                    sol.rx_id=receiver
                    sol.path_type='R'*order
                    sol.field=this_E.E
                    solved_list.append(sol)
                    
                    this_strength=np.linalg.norm(this_E.E)
                    if best['field']<this_strength:
                        best['field']=this_strength
                        best['path']=the_path
            return best
          
        def compute_diffractions(receiver,rtp,the_df,solved_list): 
            # rx=rtp.solved_receivers[receiver]
            diffractions = rtp.diffractions
            best = {'field': 0, 'path': None}
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
                    this_E=ElectromagneticField.my_diff(E_i,the_diff_path,the_diff_surfaces_ids,the_corner_points,rtp,receiver)
                    
                    sol=solved_field(the_path)
                    sol.rx_id=receiver
                    sol.path_type='R'*(order-1)+'D'
                    sol.field=this_E.E
                    solved_list.append(sol)
                    
                    # path_length=geom.path_length(the_path)
                    # the_df.loc[len(the_df.index)] = [path_length,path_length/this_E.v,np.array(this_E.E,dtype=complex),'R'*(order-1)+'D',rx,receiver,the_path[1],the_path[-2],0,0,0,0]
                    this_strength=np.linalg.norm(this_E.E)
                    if best['field']<this_strength:
                        best['field']=this_strength
                        best['path']=the_path
            return best
                
        # computes everything
        #df = pd.DataFrame(columns=['total_len','time_to_receiver','field_strength','path_type','receiver','rx_id','second_point','penultimate_point','tx_el','tx_az','rx_el','rx_az'])
        solution=[]
        for receiver in range(0,len(rtp.solved_receivers)):
            print(f"EM SOLVING RECEIVER {receiver}")
            #this_df = pd.DataFrame(columns=['total_len','time_to_receiver','field_strength','path_type','receiver','rx_id','second_point','penultimate_point','tx_el','tx_az','rx_el','rx_az']) #point transmitted, point arriving to rx
            #init_len=len(this_df.index)
            this_solved_list=[]
            rx=rtp.solved_receivers[receiver]
            best_los=compute_LOS(receiver, rtp,this_solved_list)#this_df)
            best_ref=compute_reflections(receiver, rtp,this_solved_list)# this_df)
            best_dif=compute_diffractions(receiver,rtp,this_solved_list)#this_df)
            #end_len=len(this_df.index)   
            #if len(solved_list)==0: #init_len==end_len:
                #no path at all between the RX and the TX
                #df.loc[len(df.index)] = [0,0,np.array([0,0,0],dtype=complex),'No_path_possible',rx,receiver,None,None,0,0,0,0]
            #else:
                
            #check most powerful path and align antennas to it 
            best_fields=np.array([best_los['field'],best_ref['field'],best_dif['field']])
            best_paths=[best_los['path'],best_ref['path'],best_dif['path']]
            winner=best_paths[np.argmax(best_fields)]
            print(f"winner {winner}") 
            #align antennas to best path
            tx=winner[0]
            RX_basis=antenna.align_antenna_to_point(pointA=rx,pointB=winner[-2]) #winner[-1]=rx, so we need winner[-2]
            TX_basis=antenna.align_antenna_to_point(pointA=tx,pointB=winner[1])
            
            for sol in this_solved_list:
                sol.compute_angles(RX_basis,TX_basis,tx)    
                
                #compute all angles
                # for i in range(len(this_df.index)):     
                #     second_point=this_df['second_point'][i]
                #     penultimate_point=this_df['penultimate_point'][i]
                #     rx_el,rx_az=antenna_incident_angles(RX_basis,penultimate_point-rx)
                #     tx_el,tx_az=antenna_incident_angles(TX_basis,second_point-tx)
                #     this_df['tx_el'][i]=tx_el
                #     this_df['tx_az'][i]=tx_az
                #     this_df['rx_el'][i]=rx_el
                #     this_df['rx_az'][i]=rx_az    
            solution.append(this_solved_list)
            #df=pd.concat([df,this_df],ignore_index=True)    
        #file_utils.save_df(df,save_name)#saves into the result folder
        return solution
             

class field_power:
    def harvested_power(field_strength):
        """
        given the peak to peak amplitude of the field strength arriving at the receiver in volts/meter
        returns the power extracted by the receiver antenna in Watts
        """
        lam=c/FREQUENCY
        directivity=1 #1 for a perfectly isotropic antenna
        impedance_mismatch=1 #perfect impedance matching
        polarisation_eff=1 #perfect polarisation efficiency.
        imperfections=directivity*impedance_mismatch*polarisation_eff
        A_eff=imperfections*(lam**2)/(4*pi)*RX_GAIN
        power_density=(field_strength**2)/(2*120*pi) #considering E peak to peak
        
        pr=power_density*A_eff
        return pr


    def to_db(field_power):
        """
        converts given field power in watts to dB (normalised to TX_POWER)
        """
        #avoid computing log(0)
        P=np.atleast_1d(field_power)
        ind=np.where(P!=0)
        db=np.ones(len(P))*np.NINF
        for i in ind:
            db[i]=10*np.log10(P[i]/TX_POWER) 
        return db[0] if np.isscalar(field_power) else db
         
        
    def compute_power(field_strength,STYLE=2):
        #given list of field strength, compute received power in 3 ways
        #makes it work when given np.array(1,2,3) instead of [np.array(1,2,3),...,np.array(3,4,5)]
        if isinstance(field_strength[0],np.ndarray)==False:
            field_strength=[field_strength]
        #norm of sum: takes into account interferences
        style1=np.linalg.norm(np.sum(field_strength)) #real
        style1=field_power.to_db(field_power.harvested_power(style1))
        
        #sum of individual powers
        #makes more sense to compare individual contributions
        power=0
        #TODO easier to do with:
        #amplitude=this_df["field_strength"].apply(np.linalg.norm)
        #most_powerful_path=np.argmax(amplitude)
        for i in range(len(field_strength)):   
            style2=np.linalg.norm(field_strength[i])#real
            style2=field_power.harvested_power(style2)
            power+=style2        
        style2=field_power.to_db(power)
        if STYLE==1:
            return style1
        elif STYLE==2:
            return style2
        else:
            return style1,style2


class antenna:
    def align_antenna_to_point(pointA,pointB):
        #pointA: the position of the antenna.
        #pointB: the point towards it should point.
        #points the antenna aperture towards point B from point A to point B
        #returns the antenna basis, such that:
            #new_z=(A to B) axis
            #new_x= vector in aperture and parallel to ground.
            #new_y= new_x cross new_z
            
        def parametric_from_normal_and_point(normal,point):
            #Returns the parametric equation of the plane described its normal and a point.
            #It will return all four coefficients such that: a*x + b*y + c*z + d = 0.
            normal=normal/np.linalg.norm(normal)
            a, b, c = normal
            d=-np.dot(point, normal)
            return a,b,c,d    
        
        normal=pointB-pointA
        normal=normal/np.linalg.norm(normal)
        a,b,c,d=parametric_from_normal_and_point(normal,pointA) #parametric equation of the aperture's plane  
        new_z=normal
        
        g2=pointA[1]+1 #set arbitrarily
        g3=pointA[2] #projection of x_axis on YZ is perp to Z
        g1=(-d-c*g3-b*g2)/a #G must be in aperture plane
        G=np.array([g1,g2,g3])
        new_x=G-pointA
        new_x=new_x/np.linalg.norm(new_x)
        
        new_y= np.cross(new_x,new_z)
        new_y=new_y/np.linalg.norm(new_y)
        return [new_x,new_y,new_z]

    def antenna_incident_angles(antenna_basis,ray):
        #given the antenna basis and the incoming ray 
        #returns elevation and azimuth angles of incident path in radians 
        ray=ray/np.linalg.norm(ray)
        el=np.arccos(np.dot(ray,antenna_basis[1]))#angle between incoming ray and antenna y axis
        az=np.arccos(np.dot(ray,antenna_basis[0]))#angle between incoming ray and antenna x axis
        return el,az


#TODO: fix
def EM_fields_data(df_path):
    df=file_utils.load_df(df_path)
    nreceivers=len(df['rx_id'].unique())
    for receiver in range(0,nreceivers):
        rx_coord=df.loc[df['rx_id'] == receiver]['receiver'].values[0]
        
        print(f"------------data for receiver {receiver}, with coords {rx_coord}-------------------")
        this_df=df.loc[df['rx_id'] == receiver]     
        if(this_df['path_type'].str.contains("No_path_possible").any()):
            print("No path linking TX to RX was found for this receiver")
            continue     
        elif(this_df['path_type'].str.contains("LOS").any()):
            print("there is line of sight")      
        else:
            print('NO LINE OF SIGHT')
        
        style1,style2=compute_power(this_df['field_strength'].values,STYLE=4)
        print(this_df)
        print(f"total field power style1 {style1} dB style2 {style2} dB ")
        
        #pure_reflections = np.sum(this_df[this_df['path_type'].str.contains('D|LOS',regex=True)==False]["path_power"])
        #pure_diffractions = np.sum(this_df[this_df['path_type'].str.contains('R|LOS',regex=True)==False]["path_power"])
        #mixed = np.sum(this_df[this_df['path_type'].str.contains('RD')]["path_power"])
        # print(f"from LOS {los} V/m")
        # print(f"from PURE reflections {pure_reflections} V/m")
        # print(f"from PURE diffractions {pure_diffractions} V/m")
        # print(f"from paths with reflection(s) and diffraction {mixed} V/m")
    return
    
#TODO
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
            power=compute_power(data_for_type['field_strength'].values) #receive power in dB    
            ax1.bar(x=position, height=power,width=width,color=color_for_type,label=path_type)     
            #power delay profile
            nelem=len(data_for_type['field_strength'])
            individual_powers=np.zeros(nelem)
            for j in range(nelem):
               individual_powers[j]=compute_power(data_for_type['field_strength'].values[j],STYLE=2)    
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




   
    
    
    
    
    
    
        
        
    
    
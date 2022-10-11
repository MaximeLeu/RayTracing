from raytracing import array_utils
from raytracing import geometry as geom
import numpy as np
from numpy.dual import norm

import json

from scipy.constants import c, mu_0, epsilon_0, pi
from scipy.special import fresnel
from materials_properties import DF_PROPERTIES

#n_air = 1  # Same as Matlab


n_air = 1.00026825
er = 4.44 + 1e-3j  #relative permittivity of material on which the wave reflects

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
        EF.E = np.exp(-1j*EF.k*d) * (eph + eth) / (4 * pi * d) # missing x(-jk eta) ?
        return EF

        
    def dyadic(complete_path,rtp):
        """
        Given a reflection path [TX,R1,R2,..RN,RX],[id1,...,idN] computes the dyadic 
        reflection coefficient for each path

        Parameters
        ----------
        complete_path : [TX,R1,R2,..RN,RX],[id1,...,idN]
        rtp : the post processed ray tracing problem
        Returns
        -------
        dyadic reflection coefficients [Coeff_R1,...,Coeff_RN]

        """
        def split_reflections_path(path):
            """
            split a path of multiple reflections into multiple 3 points paths
            Parameters
            ----------
            path : TYPE [[TX,R1,...RN,RX],[R1 surface id,...,RN surface id]]
                DESCRIPTION. The points describing the path of the reflection and 
                the id of the surface on which the reflection took place

            Returns [[TX,R1,P1],...,[PN,RN,RX]],[id1,...,idN]
            -------
            """  
            points=path[0]
            surface_ids=path[1]        
            paths=list()
            for i in range(0,len(points)-2):
                paths.append(points[i:i+3])
            return paths,surface_ids
        
        
        def dyadic_ref_coeff(path,surface_id):
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
            reflection_polygon= rtp.polygons[surface_id]#polygon where the reflection occured
            
            #parameters
            E=ElectromagneticField()
            
            mu_2=np.complex(reflection_polygon.properties['mu'])*mu_0
            sigma_2=np.complex(reflection_polygon.properties['sigma'])
            epsilon_2=np.complex(reflection_polygon.properties['epsilon'])*epsilon_0
            
            #wave impedance
            eta_1=np.sqrt(1j*E.w*mu_1/(sigma_1+1j*E.w*epsilon_1))
            eta_2=np.sqrt(1j*E.w*mu_2/(sigma_2+1j*E.w*epsilon_2))
            
            #propagation constant
            gamma_1=np.sqrt(1j*E.w*mu_1*(sigma_1+1j*E.w*epsilon_1))
            gamma_2=np.sqrt(1j*E.w*mu_2*(sigma_2+1j*E.w*epsilon_2))
            
            #Compute vectors from emitter to reflection point,
            #then to reflection point to receiver
      
            vectors = np.diff(path, axis=0)
            vectors, norms = geom.normalize_path(path)
            si = vectors[0, :] #normalised incident vector
            sr = vectors[1, :] #normalised reflected vector

            surface_normal=array_utils.normalize(reflection_polygon.get_normal())
            
            #incident angle and transmission angle
            theta_1=-np.arccos(np.dot(surface_normal, si)) #incident angle
            theta_2=np.arcsin(np.sin(theta_1)*gamma_1/gamma_2) #transmission angle from snell law
            
            #fresnel reflection coefficients
            r_par=(eta_2*np.cos(theta_1)-eta_1*np.cos(theta_2)) / (eta_2*np.cos(theta_1)+eta_1*np.cos(theta_2))
            r_per=(eta_2*np.cos(theta_2)-eta_1*np.cos(theta_1)) / (eta_2*np.cos(theta_2)+eta_1*np.cos(theta_1))
            
            roughness=float(reflection_polygon.properties['roughness'])
            rayleigh_criterion= (roughness > E.lam/(8*np.cos(theta_1)))
            if rayleigh_criterion:
                print("applying scattering")
                chi=np.exp(-2*E.k**2*roughness**2*np.cos(theta_1))
                r_par=r_par*chi
                r_per=r_per*chi
            
            
            #dyadic reflection coeff
            e_per=array_utils.normalize(np.cross(-si, sr))
            ei_par= array_utils.normalize(np.cross(e_per,si))
            er_par= array_utils.normalize(np.cross(e_per,-sr))
            
            R=np.array(e_per*e_per*r_per - ei_par*er_par*r_par)
            return R,norms
        
        paths,ids=split_reflections_path(complete_path)
        
        coeffs=[]
        for i in range(0,len(ids)):
            R,norms=dyadic_ref_coeff(paths[i],ids[i])
            coeffs.append([R,norms])
        return coeffs
    
    
    
        
        
    def reflect(self, reflection_path, surface_normal, surface_r_index=1/er):
        """
        Computes the electric field after a reflection
        
        :param reflection_path: Point A ----- Point B ------ Point C
         Point B is where reflection occurs -> reflection_path = [A, B, C]
         
        :param surface_normal: the normal to the surface where the reflection occurs
        
        :param surface_r_index:
            
         """
         #Compute vectors from emitter to reflection point,
         #then to reflection point to receiver
        vectors, norms = geom.normalize_path(reflection_path)
        si = vectors[0, :]
        Si = norms[0]
        sr = vectors[1, :]
        Sr = norms[1]

        # Computation of Fresnel coefficients
        c_theta = np.dot(surface_normal, -si)
        s_theta = np.sin(np.arccos(c_theta))
        n1 = n_air
        n2 = surface_r_index
        rn = n1 / n2
        rnc = rn * c_theta
        rns2 = (rn * s_theta) ** 2
        s = np.sqrt(1 - rns2)

        # ... R parallel and R perpendicular
        R_par = (rnc - s) / (rnc + s)
        R_per = (rn * s - c_theta) / (rn * s + c_theta)

        # Computation of parallel and perpendicular components
        ncsi = np.cross(surface_normal, si)
        ei_par = array_utils.normalize(np.cross(si, ncsi))
        ei_per = array_utils.normalize(np.cross(si, ei_par))

        Ei_par = np.dot(self.E, ei_par)
        Ei_per = np.dot(self.E, ei_per)

        er_par = array_utils.normalize(np.cross(ei_per, sr))

        # Reflected vector
        E = (R_per * Ei_per * ei_per + R_par * Ei_par * er_par) * np.exp(-1j*self.k*Sr) * Si / (Si + Sr)
        return ElectromagneticField(E, f=self.f, v=self.v)

    def diffract(self, diffraction_path, edge, surface_1_normal, surface_2_normal,
                 surface_r_index=1/er):
        # Compute vectors from emitter to reflection point, then to diffraction point to receiver
        vectors, norms = geom.normalize_path(diffraction_path)
        si = vectors[0, :]
        Si = norms[0]
        sd = vectors[1, :]
        Sd = norms[1]
        t = array_utils.normalize(edge)

        # sit = <si, t>
        # sdt = <sd, t>
        sit = np.dot(si, t)
        sdt = np.dot(sd, t)

        beta_0 = np.arccos(sit)

        # e = n x t
        e = np.cross(surface_1_normal, t)

        # Compute angles with si

        npi = pi + np.arccos(np.dot(surface_1_normal, surface_2_normal))
        n_diff = npi / pi
        cos_1 = np.dot(surface_1_normal, si)
        cos_2 = np.dot(surface_2_normal, si)
        sin_1 = norm(np.cross(surface_1_normal, si))
        sin_2 = norm(np.cross(surface_2_normal, si))
        a_1 = np.arctan2(sin_1, cos_1)
        a_2 = np.arctan2(sin_2, cos_2)

        # npi = a_1 + a_2 ... But a_1 can actually be a_1 or 180° - a_1,
        # same for a_2, so we need to check:
        error = np.inf
        for a_1_ in [a_1, pi - a_1]:
            for a_2_ in [a_2, pi - a_2]:
                local_error = abs(npi - a_1_ - a_2_)
                if local_error < error:
                    error = local_error
                    a_1 = a_1_
                    a_2 = a_2_
        # This will take the values of a_1 and a_2 for which equality is best satisfied

        st1 = array_utils.normalize(si - sit * t)
        st2 = array_utils.normalize(sd - sdt * t)

        st1e = np.dot(st1, e)
        st2e = np.dot(st2, e)

        phiinc = a_1
        phidiff = a_2  # TODO: correct

        # Computes the diffraction dyadic coefficients from
        # "Propagation Modelling of Low Earth-Orbit Satellite Personal
        # Communication Systems" C. Oestges

        L = Si * Sd * np.sin(beta_0)**2 / (Si + Sd)
        sm = phidiff + phiinc
        sb = phidiff - phiinc

        Npos1 = round((sb + pi) / (2 * npi))
        Nneg2 = round((sb - pi) / (2 * npi))
        Npos3 = round((sm + pi) / (2 * npi))
        Nneg4 = round((sm - pi) / (2 * npi))

        # Computation of Fresnel coefficients
        theta_0 = pi / 2 - phiinc
        theta_n = pi / 2 - min(phidiff, npi - phidiff)
        c_theta_0, s_theta_0 = np.cos(theta_0), np.sin(theta_0)
        c_theta_n, s_theta_n = np.cos(theta_n), np.sin(theta_n)

        n1 = n_air
        n2 = surface_r_index
        rn = n1 / n2
        rnc_0 = rn * c_theta_0
        rnc_n = rn * c_theta_n
        rns2_0 = (rn * s_theta_0) ** 2
        rns2_n = (rn * s_theta_n) ** 2
        s_0 = np.sqrt(1 - rns2_0)
        s_n = np.sqrt(1 - rns2_n)

        # ... R parallel and R perpendicular
        R_par_o = (rnc_0 - s_0) / (rnc_0 + s_0)
        R_par_n = (rnc_n - s_n) / (rnc_n + s_n)
        R_per_o = (rn * s_0 - c_theta_0) / (rn * s_0 + c_theta_0)
        R_per_n = (rn * s_n - c_theta_n) / (rn * s_n + c_theta_n)

        # Computes the a- and a-
        argpos1 = self.k * L * 2 * np.cos((2 * npi * Npos1 - sb) / 2) ** 2
        argneg2 = self.k * L * 2 * np.cos((2 * npi * Nneg2 - sb) / 2) ** 2
        argpos3 = self.k * L * 2 * np.cos((2 * npi * Npos3 - sm) / 2) ** 2
        argneg4 = self.k * L * 2 * np.cos((2 * npi * Nneg4 - sm) / 2) ** 2
        arg_array = np.array([argpos1, argneg2, argpos3, argneg4])
        a_array = np.sqrt(arg_array)
        fresnel_sin, fresnel_cos = fresnel(np.sqrt(pi / 2) * a_array)

        # Computes the transition function
        integrals = np.sqrt(pi / 2) * (0.5 * (1 - 1j) - (fresnel_cos - 1j * fresnel_sin))
        F = 2j * a_array * np.exp(1j * arg_array) * integrals

        # Computes the components of the diffraction coefficients
        coeff = - np.exp(-1j * pi / 4) / (2 * n_diff * np.sqrt(2 * pi * self.k) * np.sin(beta_0))
        angles = np.array([pi + sb, pi - sb, pi + sm, pi - sm])
        D = coeff * F / np.tan(angles / (2 * n_diff))
        Dh = D[0] + D[1] + R_par_n * D[2] * R_par_o * D[3]
        Ds = D[0] + D[1] + R_per_n * D[2] * R_per_o * D[3]

        """"
        Computations of the phi's and beta's direction vectors
        * phi_i = si x t normalized
        * phi_d = t x sd normalized
        * beta_i = phi_i x si
        * beta_d = phi_d x sd
        """
        phi_i = array_utils.normalize(np.cross(si, t))
        phi_d = array_utils.normalize(np.cross(sd, t))
        beta_i = np.cross(phi_i, si)
        beta_d = np.cross(phi_d, sd)

        Es = np.dot(self.E, beta_i)
        Eh = np.dot(self.E, phi_i)

        E = (-Ds * Es * beta_d - Dh * Eh * phi_d) * np.sqrt(Si / (Sd * (Si + Sd))) * np.exp(-1j * self.k * Sd)
        return ElectromagneticField(E, f=self.f, v=self.v)

def my_reflect(previous_E,reflection_path,rtp):
    """
    Compute the E field at the end of a reflection (reflection can be multiple)
    """
    print(f"reflection path {reflection_path}")
    coeffs=ElectromagneticField.dyadic(reflection_path,rtp)

    E=previous_E.E #TODO: previous_E is E at the reflection point
    for i in range(0,len(coeffs)):
        R=coeffs[i][0]
        norms=coeffs[i][1]
        Si= norms[0]#norm of incident ray
        Sr= norms[1]#norm of reflected ray   
        E=np.dot(E,R)*np.exp(-1j*previous_E.k*Sr)*Si/(Si+Sr)
    return ElectromagneticField(E, f=previous_E.f, v=previous_E.v)

def my_diff(previous_E, diffraction_data,rtp):
    """
    Parameters
    ----------
    diffraction_path : TYPE
        DESCRIPTION.
    edge : TYPE
        DESCRIPTION.
    surface_1_normal : TYPE
        DESCRIPTION.
    surface_2_normal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path,ref_surfaces_ids,diff_points,corner_points=diffraction_data
    ref_path=path[0:-1] 
    diff_path=path[0]#TODO: select 3 last points
    
    # Compute vectors from emitter to reflection point, then to diffraction point to receiver
    vectors, norms = geom.normalize_path()
    si = vectors[0, :]
    Si = norms[0] #norm of incident ray
    sd = vectors[1, :]
    Sd = norms[1] #norm of diffracted ray
    
    polygon_1 = rtp.polygons[diff_points[0]]
    polygon_2 = rtp.polygons[diff_points[1]]
    edge= np.diff(corner_points, axis=0).reshape(-1)
    
    t = array_utils.normalize(edge) #TODO: is that correct??
    
    #TODO: probably can be simplified
    phi_inc=np.cross(Si*si,t)
    phi_inc=phi_inc/np.norm(phi_inc)
    
    phi_diff=np.cross(t,sd*Sd)
    phi_diff=phi_diff/np.norm(phi_diff)
    
    beta_0_inc=np.cross(phi_inc,si*Si)
    beta_0_diff=np.cross(phi_diff,sd*Sd)
    

    #parameters
    E=ElectromagneticField()
    
    diffraction_polygon= polygon_1#TODO, what if they don't have the same properties?
    
    mu_2=np.complex(diffraction_polygon.properties['mu'])*mu_0
    sigma_2=np.complex(diffraction_polygon.properties['sigma'])
    epsilon_2=np.complex(diffraction_polygon.properties['epsilon'])*epsilon_0
    
    #wave impedance
    eta_1=np.sqrt(1j*E.w*mu_1/(sigma_1+1j*E.w*epsilon_1))
    eta_2=np.sqrt(1j*E.w*mu_2/(sigma_2+1j*E.w*epsilon_2))
    
       
    #fresnel reflection  #TODO: are the angles correct?
    r_par=(eta_2*np.cos(beta_0_inc)-eta_1*np.cos(beta_0_diff)) / (eta_2*np.cos(beta_0_inc)+eta_1*np.cos(beta_0_diff))
    r_per=(eta_2*np.cos(beta_0_diff)-eta_1*np.cos(beta_0_inc)) / (eta_2*np.cos(beta_0_diff)+eta_1*np.cos(beta_0_inc))
    
    
    #dyadic diffraction coefficient----------------------
    #From "Propagation Modelling of Low Earth-Orbit Satellite Personal
    # Communication Systems" C. Oestges
    sin_beta_0=np.norm(np.cross(Si*si,t))
    alpha=1 #interior angle of the wedge #TODO
    n=2-alpha/np.pi
    k=previous_E.k
    
    #efficiently computs the transition function, as demonstrated in
    #https://eertmans.be/research/utd-transition-function/
    def F(x):
        factor = np.sqrt(np.pi / 2)
        sqrtx = np.sqrt(x)
        S, C = fresnel(sqrtx / factor)
        transition_function=2j * sqrtx* np.exp(1j * x)* (factor * ((1 - 1j) / 2 - C + 1j * S))
        return transition_function
    
    def A(x,sign):
        match sign:
            case "minus":
                N=round((-pi+x)/(2*n*pi))
            case "plus":
                N=round((pi+x)/(2*n*pi))
            case _:
                raise ValueError('sign should either be plus or minus')                
        result=2*(np.cos(n*pi*N-x/2))**2
        return result
    
    common_factor=-np.exp(-1j*pi/4)/(2*n*np.sqrt(2*pi*k)*sin_beta_0)
    L=(sin_beta_0**2)*Si*Sd/(Si+Sd) #TODO consider plane or spherical wave front???
    
    #components of the diffraction coefficient
    cot1=1/np.tan((pi+(phi_diff-phi_inc))*1/(2*n))
    cot2=1/np.tan((pi-(phi_diff-phi_inc))*1/(2*n))
    cot3=1/np.tan((pi+(phi_diff+phi_inc))*1/(2*n))
    cot4=1/np.tan((pi-(phi_diff+phi_inc))*1/(2*n))
    
    F1=F(k*L*A(phi_diff-phi_inc,"plus"))
    F2=F(k*L*A(phi_diff-phi_inc,"minus"))
    F3=F(k*L*A(phi_diff+phi_inc,"plus"))
    F4=F(k*L*A(phi_diff+phi_inc,"minus"))
    
    D1=common_factor*cot1*F1
    D2=common_factor*cot2*F2
    D3=common_factor*cot3*F3
    D4=common_factor*cot4*F4
    
    D_per=D1+D2+r_per*(D3+D4)
    D_par=D1+D2+r_par*(D3+D4)
    
    #TODO eq8.31 of UTD book slightly different from claude thesis, check which is correct
    D=-D_per*beta_0_inc*beta_0_diff - D_par*phi_inc*phi_diff #TODO
        
      
    #field computation----------------------------------- 
    E_0=previous_E.E #E reaching the first interaction point
    E_i=my_reflect(E_0,[ref_path,ref_surfaces_ids],rtp) #E reaching the diffraction point
    
    E_i.E=E_i.E*D*np.sqrt(Si/(Sd*(Sd+Si)))*np.exp(-1j*E_i.k*Sd) #final field   
    return E_i

def my_field_computation(rtp):
    #TODO, given a complete path between TX and RX with whatever in between,
        #compute the field at the end of the path
    #TODO in another function, compute the field of all the paths and sum the to get the final field
    los = rtp.los
    reflections = rtp.reflections
    diffractions = rtp.diffractions
    polygons = rtp.polygons
     
    # computes field for each reflection and diffraction
    for receiver in range(0,len(rtp.place.set_of_points)):
        #compute field resulting from all reflections:         
        reflections_field=ElectromagneticField()
        reflections_field.E=0    
        for order in reflections[receiver]:
            for path in range(0,len(reflections[receiver][order])):
                the_path=reflections[receiver][order][path]
                tx =the_path[0][0] #first point of the reflection
                first_interaction=the_path[0][1] #second point of the path
                assert(np.array_equal(rtp.emitter[0],tx))
                E_initial=ElectromagneticField.from_path([tx,first_interaction])
                this_E=my_reflect(E_initial,the_path,rtp)
                reflections_field.E+=this_E.E
        print(f"field resulting from all reflections to receiver {receiver} is {reflections_field.E}")        
                
    #     #compute all diffractions
        diffractions_field=ElectromagneticField()
        diffractions_field.E=0
        for order in range(0,len(diffractions[receiver])):     
            for path in range(0,len(diffractions[receiver][order])):
                the_path=reflections[receiver][order][path]
                tx =the_path[0][0] #first point of the reflection
                first_interaction=the_path[0][1] #second point of the path
                E_initial=ElectromagneticField.from_path([tx,first_interaction])
                
                this_E=my_diff(E_initial,path,rtp)
                diffractions_field.E+=this_E.E
        print(f"field resulting from all diffractions to receiver {receiver} is {diffractions_field.E}")  
    total_field=ElectromagneticField()
    total_field.E=diffractions_field.E+reflections_field.E
    print(f"Total field resulting to receiver {receiver} is {total_field.E}")  
    return total_field


def compute_field_from_solution(rtp, output):
    """
    Computes the electromagnetic field from a ray tracing problem solution, and writes
    all the different EM fields in a json file.
    """
    los = rtp.los
    reflections = rtp.reflections
    diffractions = rtp.diffractions
    polygons = rtp.polygons

    out = {}

    for r, receiver in enumerate(rtp.receivers):

        out[r] = dict(receiver_position=receiver.tolist(), los=[], reflections=[], diffractions=[])

        E = 0
        for path in los[r]:
            rt = ElectromagneticField.from_path(path)

            out[r]["los"].append(dict(path=path.tolist(), E_real=np.real(rt.E).tolist(), E_imag=np.imag(rt.E).tolist()))

            E += np.real(rt.E.conj() @ rt.E.T)

        for order, paths in reflections[r].items():

            for path, indices in paths:
                out[r]["reflections"].append(dict(order=order, path=path.tolist(), E_real=[], E_imag=[]))

                rt = ElectromagneticField.from_path(path[:2, :])
                for i, index in enumerate(indices):
                    polygon = polygons[index]
                    normal = polygon.get_normal()
                    rt = rt.reflect(path[i:i+3, :], normal)
                    out[r]["reflections"][-1]["E_real"].append(np.real(rt.E).tolist())
                    out[r]["reflections"][-1]["E_imag"].append(np.imag(rt.E).tolist())

                E += np.real(rt.E.conj() @ rt.E.T)

        for order, paths in diffractions[r].items():
            for path, indices, (diff_p1, diff_p2), edge in paths:
                out[r]["diffractions"].append(dict(order=order, path=path.tolist(), E_real=[], E_imag=[]))
                rt = ElectromagneticField.from_path(path[:2, :])
                for i, index in enumerate(indices):
                    polygon = polygons[index]
                    normal = polygon.get_normal()
                    rt = rt.reflect(path[i:i+3, :], normal)
                    out[r]["diffractions"][-1]["E_real"].append(np.real(rt.E).tolist())
                    out[r]["diffractions"][-1]["E_imag"].append(np.imag(rt.E).tolist())

                # Last operation is diffraction
                polygon_1 = polygons[diff_p1]
                polygon_2 = polygons[diff_p2]
                normal_1 = polygon_1.get_normal()
                normal_2 = polygon_2.get_normal()

                # Edge becomes a vector
                edge = np.diff(edge, axis=0).reshape(-1)

                rt = rt.diffract(
                    path[-3:, :], edge,
                    normal_1, normal_2,
                )
                out[r]["diffractions"][-1]["E_real"].append(np.real(rt.E).tolist())
                out[r]["diffractions"][-1]["E_imag"].append(np.imag(rt.E).tolist())

                E += np.real(rt.E.conj() @ rt.E.T)

        with open(output, "w") as f:
            json.dump(out, f, sort_keys=True, indent=4)

        # Power
        # EE = conj(E) @ E^T (hermitian transpose)
        print("E", E)
        #EE = E.conj() @ E.T
        EE = np.real(E)
        print("EE", EE)

        dBuV = 20 * np.log10(1e6 * np.sqrt(EE / 2))
        print(dBuV, "dBuV at position", receiver)

       



    
    
    
    
    
    
    
    
    
    
    
        
        
    
    
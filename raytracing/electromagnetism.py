from raytracing import array_utils
from raytracing import geometry as geom
import numpy as np
from numpy.dual import norm

import json

from scipy.constants import c, mu_0, epsilon_0, pi
from scipy.special import fresnel


#n_air = 1  # Same as Matlab

n_air = 1.00026825
er = 4.44 + 1e-3j  #relative permittivity of material on which the wave reflects
Z0=np.sqrt((mu_0/epsilon_0)) #free space impedance

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

    #TODO: my function
    def compute_E(path_of_points):
        """
        Compute the electric field at the end of the path of points describing the path from TX to RX
        :param path_of_points: TX-reflection or diffraction points-RX
        :type path_of_points: numpy.ndarray(shapely.geometry.Points) *shape=(N)
        """
        
        def compute_antenna_radiation(path):
            """
            Given a direct path between TX and the next obstruction, 
            computes the electric field at the end of the path using the far field approximation  
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
            
        def compute_path_loss(path):
            """
            Computes the path loss between two points in dB
            :param path_of_points: two points
            :type path_of_points: numpy.ndarray(shapely.geometry.Points) *shape=(2)
            """
            _, d = geom.normalize_path(path)
            EF = ElectromagneticField()
            n=2#path loss exponent
            L=10*np.log10((4*pi*d/EF.lam)**n)
            return L
            
        def compute_reflection_coeff(reflection_path,Z_1,Z_2,surface_normal):
            """
            Computes the electric field after a reflection
            :param reflection_path: Point A ----- Point B ------ Point C
             Point B is where reflection occurs -> reflection_path = [A, B, C]
             
            :param Z_1: characteristic impedance of the medium point A is in
            :param Z_2: characteristic impedance of the surface of reflection (point B)
            :param surface_normal: the normal to the reflection surface
            """
            
            
            #Compute vectors from emitter to reflection point,
            #then to reflection point to receiver
            vectors, norms = geom.normalize_path(reflection_path)
            si = vectors[0, :] #normalised incident vector
            sr = vectors[1, :] #normalised reflected vector
            
            
            critical_angle= 0#TODO
            
            theta_1=np.arccos(np.dot(surface_normal, -si)) #incident angle
            theta_2= 1#transmission angle TODO
            
            #fresnel
            r_par=(Z_2*np.cos(theta_1)-Z_1*np.cos(theta_2)) / (Z_2*np.cos(theta_1)+Z_1*np.cos(theta_2))
            r_per=(Z_2*np.cos(theta_2)-Z_1*np.cos(theta_1)) / (Z_2*np.cos(theta_2)+Z_1*np.cos(theta_1))
            
            
            return

        def compute_diffraction_coeff(diffraction_path):
            return
        
        
        return
        
    
    
    def dyadic_ref_coeff(path,mu_1,mu_2,sigma_1,sigma_2,epsilon_1,epsilon_2):
       
      #fresnel
        E=ElectromagneticField()
        #wave impedance
        eta_1=np.sqrt(1j*E.w*mu_1/(sigma_1+1j*E.w*epsilon_1))
        eta_2=np.sqrt(1j*E.w*mu_2/(sigma_2+1j*E.w*epsilon_2))
        
        #propagation constant
        gamma_1=np.sqrt(1j*E.w*mu_1*(sigma_1+1j*E.w*epsilon_1))
        gamma_2=np.sqrt(1j*E.w*mu_2*(sigma_2+1j*E.w*epsilon_2))
        
        
        #Compute vectors from emitter to reflection point,
        #then to reflection point to receiver
        vectors, norms = geom.normalize_path(path)
        si = vectors[0, :] #normalised incident vector
        sr = vectors[1, :] #normalised reflected vector

        #required vectors
        surface_normal=1 #TODO
        e_per=1 #TODO
        e_par_i= 1 #TODO
        e_par_r= 1 #TODO


        #incident and transmission angles
        theta_1=np.arccos(np.dot(surface_normal, -si))
        theta_2=np.arcsin(np.sin(theta_1)*gamma_1/gamma_2)
        
        #fresnel reflection coefficients
        r_par=(eta_2*np.cos(theta_1)-eta_1*np.cos(theta_2)) / (eta_2*np.cos(theta_1)+eta_1*np.cos(theta_2))
        r_per=(eta_2*np.cos(theta_2)-eta_1*np.cos(theta_1)) / (eta_2*np.cos(theta_2)+eta_1*np.cos(theta_1))
        
        #dyadic reflection coeff
        R=e_per*e_per*r_per - e_par_i*e_par_r*r_par
        
        return R
        
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

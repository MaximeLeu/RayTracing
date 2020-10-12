from radarcoverage import array_utils
from radarcoverage import geometry as geom
import numpy as np
from numpy.dual import norm

from scipy.constants import c, mu_0, epsilon_0, pi
from scipy.special import fresnel

n_air = 1.00026825
c = 3e8
er = 4.44 + 1e-3j


class ElectromagneticField:

    def __init__(self, E=np.array([1, 0, 0], dtype=complex), f=1.8e9, v=c):
        self.E = E
        self.f = f
        self.v = v
        self.k = 2 * pi * f / v

    @staticmethod
    def from_path(path):
        vector, d = geom.normalize_path(path)
        phi, theta, _ = geom.cartesian_to_spherical(vector * d)
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
        EF.E = np.exp(-1j*EF.k*d) * (eph + eth) / (4 * pi * d)
        return EF

    def reflect(self, reflection_path, surface_normal, surface_r_index):
        # Compute vectors from emitter to reflection point, then to reflection point to receiver
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
        R_par = np.abs((rnc - s) / (rnc + s)) ** 2
        R_per = np.abs((rn * s - c_theta) / (rn * s + c_theta))

        # Computation of parallel and perpendicular components
        ncsi = np.cross(surface_normal, si)
        ei_par = array_utils.normalize(np.cross(si, ncsi))
        ei_per = array_utils.normalize(np.cross(si, ei_par))

        Ei_par = np.dot(self.E, ei_par)
        Ei_per = np.dot(self.E * ei_per)

        er_par = array_utils.normalize(np.cross(ei_per, sr))

        # Reflected vector
        return (R_per * Ei_per * ei_per + R_par * Ei_par * er_par) * np.exp(-1j*self.k*Sr) * Si / (Si + Sr)

    def diffract(self, diffraction_path, edge, surface_1_normal, surface_2_normal):
        # Compute vectors from emitter to reflection point, then to diffraction point to receiver
        vectors, norms = geom.normalize_path(diffraction_path)
        si = vectors[0, :]
        Si = norms[0]
        sd = vectors[1, :]
        Sd = norms[1]
        t = array_utils.normalize(edge)

        sit = np.dot(si, t)
        sdt = np.dot(sd, t)
        e = np.cross(surface_1_normal, t)

        # Compute angles with si

        npi = pi + np.arccos(np.dot(surface_1_normal, surface_2_normal))
        cos_1 = np.dot(surface_1_normal, si)
        cos_2 = np.dot(surface_2_normal, si)
        sin_1 = norm(np.cross(surface_1_normal, si))
        sin_2 = norm(np.cross(surface_2_normal, si))
        a_1 = np.arctan2(sin_1, cos_1)
        a_2 = np.arctan2(sin_2, cos_2)

        # npi = a_1 + a_2 ... But a_1 can actually be a_1 or 180° - a_1, same for a_2, so we need to check:
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

        L = Si * Sd * np.sin(beta_0)**2 / (Si + Sd)

        """
        Computes the components of the diffraction coefficients
        """
        coeff = - np.exp(-1j * pi / 4) / (2 * n_diff * np.sqrt(2 * pi * self.k) * np.sin(beta_0))
        D1 = coeff * Fpos1 *

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

        return (-Ds * Es * beta_d - Dh * Eh * phi_d) * np.sqrt(Si / (Sd * (Si + Sd))) * np.exp(-1j * self.k * Sd)

    
        
if __name__ == '__main__':

    c = 3e8
    f = 1.8e9
    k = 2*pi*f/c

    p1 = np.array([5, 12, 65])
    p2 = np.array([65, 12, 65])

    ElectromagneticField.from_path(np.row_stack([p1, p2]))

    d = norm(p1 - p2)
    vec = (p2 - p2)/d
    phi, theta, _ = geom.cartesian_to_spherical(vec)
    theta = pi/2 - theta
    if theta < 0:
        theta += 2*pi
    theta = theta % 2*pi
    phi = phi % 2*pi
    print(phi, theta)
    eph = np.array([-np.sin(phi), np.cos(phi), 0])
    eth = np.array([np.cos(theta)*np.cos(phi), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    Ei = (np.exp(-1j*k*d) / (4*pi*d)) * (eph + eth)
    print(Ei)

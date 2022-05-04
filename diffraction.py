from numpy import *
import numpy as np
import scipy.special as sc
from scipy.constants import c, pi

def a_plus(n, angle):
    N = np.round((pi + angle) / (2 * pi * n))
    return 2 * np.cos((2 * n * pi * N - angle) / 2) ** 2


def a_minus(n, angle):
    N = np.round((-pi + angle) / (2 * pi * n))
    return 2 * cos((2 * n * pi * N - angle) / 2) ** 2


def cot(x):
    return 1 / tan(x)


def F(x):

    factor = np.sqrt(np.pi / 2)
    sqrtx = np.sqrt(x)

    S, C = sc.fresnel(sqrtx / factor)

    return (
        2j
        * sqrtx
        * np.exp(1j * x)
        * (factor * ((1 - 1j) / 2 - C + 1j * S))
        # We changed the parenthesis so that
        # \sqrt{pi/2} now multiplies C and S
    )


def D_s_h(n, k, beta_0, phi, phi_p, Li, Lrn, Lro):

    factor = -exp(-1j * pi / 4) / (2 * n * sqrt(2 * pi * k) * sin(beta_0))

    D1 = (
        factor
        * cot((pi + (phi - phi_p)) / (2 * n))
        * F(k * Li * a_plus(n, phi - phi_p))
    )
    D2 = (
        factor
        * cot((pi - (phi - phi_p)) / (2 * n))
        * F(k * Li * a_minus(n, phi - phi_p))
    )
    D3 = (
        factor
        * cot((pi + (phi + phi_p)) / (2 * n))
        * F(k * Lrn * a_plus(n, phi + phi_p))
    )
    D4 = (
        factor
        * cot((pi - (phi + phi_p)) / (2 * n))
        * F(k * Lro * a_minus(n, phi + phi_p))
    )

    return D1 + D2 - D3 - D4, D1 + D2 + D3 + D4


class Wave:
    def __init__(self, f=1e9):
        self.f = f
        self.l = c / self.f
        self.k = 2 * pi / self.l


class Edge:
    def __init__(self, direction, normals):
        self.normals = np.asarray(normals, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.angle = pi - arccos(dot(self.normals[0, :], self.normals[1, :]))
        
        #print("Angle", np.rad2deg(self.angle))


class Vector:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)
        self.len = np.linalg.norm(self.vector)
        self.direction = self.vector / self.len


def unit(vector):
    return Vector(vector).direction


def reflected_field(field, path, normal, wave):
    S = path[0, :]
    R = path[1, :]
    O = path[2, :]

    k = wave.k
    n = unit(normal)

    s_i = Vector(R - S)
    s_r = Vector(O - R)
    
    e_per_i = unit(cross(s_i.direction, normal))
    e_par_i = unit(cross(e_per_i, s_i.direction))
    
    e_per_r = e_per_i
    e_par_r = unit(cross(e_per_r, s_r.direction))
    
    # For PEC
    tau_per = tau_par = -1
    
    Ei = array([dot(field, e_per_i), dot(field, e_per_r)])
    
    D = array([[tau_per, 0], [0, -tau_par]], dtype=complex64)
    
    rho = s_i.len
    s = s_r.len
    factor = rho / (s + rho) * np.exp(-1j * k * s)
    
    Ed = factor * (D @ Ei)

    return Ed[0] * e_per_r + Ed[1] * e_par_r


def diffracted_field(field, path, edge, wave):
    S = path[0, :]
    D = path[1, :]
    O = path[2, :]

    k = wave.k
    alpha = edge.angle
    e = edge.direction

    s_i = Vector(D - S)
    s_d = Vector(O - D)
    
    # TODO: remove
    theta = np.arctan2(path[2, -1], path[2, 0])

    n = (2 * pi - alpha) / pi
    beta_0 = arcsin(np.linalg.norm(cross(s_i.direction, e)))

    if dot(s_i.direction, -edge.normals[0]) >= dot(s_i.direction, -edge.normals[1]):
        n_o = edge.normals[0]  # Normal to outer surface
    else:
        n_o = edge.normals[1]

    phi_i = unit(-cross(e, s_i.direction))
    beta_0_i = unit(+cross(phi_i, s_i.direction))

    phi_d = unit(cross(e, s_d.direction))
    beta_0_d = unit(cross(phi_d, s_d.direction))

    s_t_i = unit(s_i.direction - dot(s_i.direction, e) * e)
    s_t_d = unit(s_d.direction - dot(s_d.direction, e) * e)

    t = cross(n_o, e)
    

    phi_i_angle = pi - (pi - arccos(clip(dot(-s_t_i, t), -1, 1))) * sign(dot(-s_t_i, n_o))
    phi_d_angle = pi - (pi - arccos(clip(dot(+s_t_d, t), -1, 1))) * sign(dot(+s_t_d, n_o))
    
    # Diffraction

    Li = Lro = Lrn = s_d.len * sin(beta_0) ** 2

    D_s, D_h = D_s_h(n, k, beta_0, phi_d_angle, phi_i_angle, Li, Lrn, Lro)
    

    Ei = array([dot(field, beta_0_i), dot(field, phi_i)])

    D = array([[-D_s, 0], [0, -D_h]], dtype=complex64)

    rho = s_i.len
    s = s_d.len
    factor = np.sqrt(rho / (s * (s + rho))) * np.exp(-1j * k * s)

    Ed = factor * (D @ Ei)

    return Ed[0] * beta_0_d + Ed[1] * phi_d
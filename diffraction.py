from numpy import *
import numpy as np
import scipy.special as sc

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


def length(vec):
    return np.linalg.norm(vec)


def unit(vec):
    return vec / length(vec)


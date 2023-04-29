import numpy as np


def prop_coeff(r):
    """
    Calculates coil reduction ratio
    ---------------
    @param r: List of radii
    @return: Coil reduction ratio
    """
    prop = []

    for i in range(len(r) - 1):
        prop.append(np.sum(r[i+1]) / np.sum(r[i]))

    return prop

def Coil_resistance(material, l, d, nu):
    """
    Calculates the coil resistance
    ---------------
    @param material: Contour material
    @param l: Contour length
    @param d: Diameter of the conductor cross section
    @param nu: The frequency of the current in the contour
    @return: Coil resistance
    """

    Epsilon0 = 8.85419e-12
    omega = 2*np.pi*nu
    c = 299792458

    ro = {
        'Silver': 0.016,
        'Copper': 0.0168,
        'Gold': 0.024,
        'Aluminum': 0.028,
        'Tungsten': 0.055}

    ro[material] = ro[material]*1e-6

    delta = c * np.sqrt((2*Epsilon0*ro[material]) / omega)

    S_eff = np.pi*(d / 2)**2 - np.pi * (d/2 - delta) **2

    R = (ro[material] * (l / S_eff))

    return R

def resistance_contour(prop, max_r, material, d, nu):
    """
    Calculates the contour resistance
    ---------------
    @param prop: Coil reduction ratio
    @param max_r: Maximum length of the coil
    @param material: Contour material
    @param d: Diameter of the conductor cross section
    @param nu: The frequency of the current in the contour
    @return: Contour resistance
    """
    coils = [max_r]
    for i in prop:
        coils.append(i*max_r)
    R = []
    for i in coils:
        R.append(Coil_resistance(material, 2 * np.pi * i, d, nu))
    return (np.sum(list(map(lambda x: x ** (-1), R)))) ** (-1)

import numpy as np


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

def resistance_contour(l, material, d, nu):
    """
    Calculates the contour resistance
    ---------------
    @param l: Array of lengths of parallel connected coils
    @param material: Contour material
    @param d: Diameter of the conductor cross section
    @param nu: The frequency of the current in the contour
    @return: Contour resistance
    """
    Resistance_coil = []
    for i in l:
        Resistance_coil.append(Coil_resistance(material, l, d, nu))
    return (np.sum(list(map(lambda x: x ** (-1), Resistance_coil)))) ** (-1)


def length_circular_coils(coils):
    """
    Calculates the length of each sequentially connected turn in a circular contour
    ---------------
    @param coils: Multidimensional array with the radii of the turns
    @return l: Array with lengths of sequentially connected turns
    """
    l = []
    for coil in coils:
        l.append(2 * np.pi * np.sum(coil))
    return l


def length_square_coils(coils):
    """
    Calculates the length of each sequentially connected turn in a rectangular contour
    ---------------
    @param coils: Multidimensional array with the lengths of the sides of the turns
    @return l: Array with lengths of sequentially connected turns
    """
    l = []
    for coil in coils:
        lenght_coil = 0
        for i in coil:
            lenght_coil += np.sum(i)
        l.append(lenght_coil)
    return l


def length_piecewise_linear_coils(coils):
    """
    Calculates the length of each sequentially connected turn in a piecewise linear contour
    ---------------
    @param coils: Multidimensional array with the coordinates of the turns
    @return l: Array with lengths of sequentially connected turns
    """
    l = []
    for coil in coils:
        lenght_coil = 0
        for i in range(len(coil)):
            try:
                lenght_coil += np.sqrt((coil[i][0] - coil[i+1][0])**2 + (coil[i][1] - coil[i+1][1])**2)
            except IndexError:
                lenght_coil += np.sqrt((coil[0][0] - coil[i][0]) ** 2 + (coil[0][1] - coil[i][1]) ** 2)
        l.append(lenght_coil)
    return l
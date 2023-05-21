import numpy as np


def transposition(xv, yv, zv):
    cp = len(xv)

    xv_T = np.zeros((cp, cp, cp))
    yv_T = np.zeros((cp, cp, cp))
    zv_T = np.zeros((cp, cp, cp))

    for i in range(cp):
        xv_T[:, :, i] = xv[:, :, i].T
        yv_T[:, :, i] = yv[:, :, i].T
        zv_T[:, :, i] = zv[:, :, i].T

    return xv_T, yv_T, zv_T


def prop_coeff(R):
    """
    Calculates the radius reduction factor
    ---------------
    """
    R.sort()
    R.reverse()
    prop = []
    for i in range(len(R)-1):
        prop.append(R[i]/max(R))

    return prop

def Radii_in_coords(R, coords_max):
    """
    Converts radii to vertex coordinates of a piecewise linear coil
    ---------------
    """
    prop = prop_coeff(R)
    list_of_coords = [coords_max]
    for k in prop:
        new_coords = []
        for point in coords_max:
            new_coords.append([point[0] * k, point[1] * k])
        list_of_coords.append(new_coords)
        new_coords.clear()
    return list_of_coords


def Bz_segment(start_point, end_point, g, I, spacing, cp):
    """
    Calculates z-component of B field of single-segment
    ---------------
    @param x1, y1: The beginning of the segment
    @param x2, y2: The end of the segment
    @param g: Length of the largest segment
    @param I: Current in the contour
    @param spacing: Spacing between coil and the calculation domain boundary
    @param cp: Calculation domain points
    @return: Z-component of B field of single-segment
    """
    x1, y1 = start_point[0], start_point[1]
    x2, y2 = end_point[0], end_point[1]
    mu0 = np.pi * 4e-7
    C = mu0 * I / (4 * np.pi)

    calc_radius = g * spacing
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)

    xv, yv, zv = transposition(xv, yv, zv)

    if x1 != x2 and y1 != y2:

        k = (y2 - y1) / (x2 - x1)
        b = y2 - k*x2

        alpha = np.sqrt(k**2 + 1)
        betta = (xv + k*yv - b*k) / (alpha**2)
        gamma = yv - (k*xv + b)
        delta = zv**2 + (gamma / alpha)**2

        Bz_segment_1 = C * ((x1 - betta) * gamma) / (delta * np.sqrt((alpha * (x1 - betta))**2 + delta))
        Bz_segment_2 = C * ((x2 - betta) * gamma) / (delta * np.sqrt((alpha * (x2 - betta))**2 + delta))

        return Bz_segment_2 - Bz_segment_1

    elif x1 == x2 and y1 != y2:

        alpha = zv**2 + (x1 - xv)**2

        Bz_segment_1 = C * ((x1 - xv) * (y1 - yv)) / (alpha * np.sqrt((y1 - yv)**2 + alpha))
        Bz_segment_2 = C * ((x1 - xv) * (y2 - yv)) / (alpha * np.sqrt((y2 - yv)**2 + alpha))

        return Bz_segment_2 - Bz_segment_1

    elif x1 != x2 and y1 == y2:

        alpha = zv ** 2 + (y1 - yv) ** 2

        Bz_segment_1 = C * ((yv - y1) * (x1 - xv)) / (alpha * np.sqrt((x1 - xv) ** 2 + alpha))
        Bz_segment_2 = C * ((yv - y1) * (x2 - xv)) / (alpha * np.sqrt((x2 - xv) ** 2 + alpha))

        return Bz_segment_2 - Bz_segment_1

def Bz_piecewise_linear_contour_single(coords,  I, spacing, cp, direction):
    """
    Calculates Bz field of piecewise linear coil
    ---------------
    @param coords: Coordinates of the contour corners
    @param I: Current in the contour
    @param spacing: Spacing between coil and the calculation domain boundary
    @param cp: Calculation domain points
    @param direction: The direction of the current along the contour. If the current flows clockwise, then by default this value is True
    @return: Z-component B of the field of single coil
    """
    if not direction:
        I = -I

    l = []
    for i in range(len(coords)):
        l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))

    g = np.amax(l)
    I = np.sqrt(2)*I

    Bz_piecewise_linear_contour_single = np.zeros((cp, cp, cp))
    for i in range(len(coords)):
        try:
            Bz_piecewise_linear_contour_single += Bz_segment(coords[i], coords[i + 1], g, I, spacing, cp)
        except IndexError:
            Bz_piecewise_linear_contour_single += Bz_segment(coords[i], coords[0], g, I, spacing, cp)

    return Bz_piecewise_linear_contour_single


def Bz_piecewise_linear_contour(R, coords,  I, spacing, cp, direction=True):
    """
    Calculates the Bz field for a piecewise linear contour
    ---------------
    @param R: Set of radii
    @param coords: Coordinates of the largest turn
    @param I: Current in the contour
    @param spacing: Spacing between coil and the calculation domain boundary
    @param cp: Calculation domain points
    @param direction: The direction of the current along the contour. If the current flows clockwise, then by default this value is True
    @return: Z-component B of the field of a piecewise linear contour
    """
    list_of_coords = Radii_in_coords(R, coords)

    Bz_piecewise_linear_contour = np.zeros([cp, cp, cp])

    for coil in list_of_coords:
        Bz_piecewise_linear_contour += Bz_piecewise_linear_contour_single(coil, I, spacing, cp, direction)

    return Bz_piecewise_linear_contour
import numpy as np
import math
from scipy.special import ellipk,ellipkm1, ellipe


def transposition(xv, yv, zv):
    cp = len(xv)

    xv_T = np.zeros((cp, cp, cp))
    yv_T = np.zeros((cp, cp, cp))
    zv_T = np.zeros((cp, cp, cp))

    for i in range(cp):
        xv_T[i] = xv[:, :, i].T
        yv_T[i] = yv[:, :, i].T
        zv_T[i] = zv[:, :, i].T

    return xv_T, yv_T, zv_T


def prop_coeff(R):
    """
    Calculates the radius reduction factor
    ---------------
    """
    R.sort()
    R.reverse()
    prop = []
    max_r = max(R)
    for i in range(len(R)):
        prop.append(R[i]/max_r)

    return prop


def Radii_in_sides_square(R, X_side, Y_side):
    """
    Converts radii to sides of a rectangular coil
    ---------------
    """
    prop = prop_coeff(R)
    X_sides, Y_sides = [X_side], [Y_side]
    for k in prop:
        X_sides.append(X_side * k)
        Y_sides.append(Y_side * k)

    return X_sides, Y_sides


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
        betta = (xv + k*yv - b) / (alpha**2)
        gamma = yv - (k*xv + b)
        delta = zv**2 + (gamma / alpha)**2

        Bz_segment_1 = C * (abs(x1 - betta) * gamma) / (delta * np.sqrt((alpha * (x1 - betta))**2 + delta))
        Bz_segment_2 = C * (abs(x2 - betta) * gamma) / (delta * np.sqrt((alpha * (x2 - betta))**2 + delta))

        return Bz_segment_2 - Bz_segment_1

    elif x1 == x2 and y1 != y2:

        alpha = zv**2 + (x1 - xv)**2

        Bz_segment_1 = C * ((x1 - xv) * abs(y1 - yv)) / (alpha * np.sqrt((y1 - yv)**2 + alpha))
        Bz_segment_2 = C * ((x1 - xv) * abs(y2 - yv)) / (alpha * np.sqrt((y2 - yv)**2 + alpha))

        return Bz_segment_2 - Bz_segment_1

    elif x1 != x2 and y1 == y2:

        alpha = zv ** 2 + (y1 - yv) ** 2

        Bz_segment_1 = C * ((yv - y1) * abs(x1 - xv)) / (alpha * np.sqrt((x1 - xv) ** 2 + alpha))
        Bz_segment_2 = C * ((yv - y1) * abs(x2 - xv)) / (alpha * np.sqrt((x2 - xv) ** 2 + alpha))

        return Bz_segment_2 - Bz_segment_1


def Bz_piecewise_linear_contour_single(coords,  I, spacing, cp, g, direction):
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
    I = np.sqrt(2) * I

    if not direction:
        I = -I

    Bz_piecewise_linear_contour_single = np.zeros((cp, cp, cp))
    for i in range(len(coords) - 1):
        Bz_piecewise_linear_contour_single += Bz_segment(coords[i], coords[i + 1], g, I, spacing, cp)

    Bz_piecewise_linear_contour_single += Bz_segment(coords[len(coords) - 1], coords[0], g, I, spacing, cp)

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

    l = []
    for i in range(len(coords)):
        l.append(np.sqrt((coords[i][0]) ** 2 + (coords[i][1]) ** 2))

    g = max(l)

    Bz_piecewise_linear_contour = np.zeros((cp, cp, cp))

    for coil in list_of_coords:
        Bz_piecewise_linear_contour += Bz_piecewise_linear_contour_single(coil, I, spacing, cp, g, direction)

    return Bz_piecewise_linear_contour


def Bz_circular_single(r_max, a, I, spacing, cp):
    """
    Calculates the Bz field of a single circular coil
    ---------------
    @param a: Turn radius
    @param I: Current in the contour
    @param spacing: Spacing between coil and the calculation domain boundary
    @param cp: Calculation domain points
    @return: Z-component B of the field of a single circular coil
    """
    mu0 = np.pi * 4e-7
    calc_radius = r_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid

    xv, yv, zv = transposition(xv, yv, zv)

    ro = np.sqrt(xv ** 2 + yv ** 2)
    r = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2)
    C = mu0 * I / np.pi
    alpha = np.sqrt(a ** 2 + r ** 2 - 2 * a * ro)
    beta = np.sqrt(a ** 2 + r ** 2 + 2 * a * ro)
    k = np.sqrt(1 - alpha ** 2 / beta ** 2)

    Bz = C / (2 * alpha ** 2 * beta) * ((a ** 2 - r ** 2) * ellipe(k ** 2) + alpha ** 2 * ellipk(k ** 2))

    return Bz


def Bz_circular_contour(R, I, spacing, cp):
    """
    Calculates the Bz field for a circular contour
    ---------------
    """
    I = np.sqrt(2) * I
    R.sort()
    R.reverse()
    Bz_circular_contour = np.zeros((cp, cp, cp))

    r_max = max(R)

    for r in R:
        Bz_circular_contour += Bz_circular_single(r_max, r, I, spacing, cp)


    return Bz_circular_contour


def Bz_square_single(m, n, I, spacing, cp, max_side):
    """
    Calculates the Bz field of a single square coil
    ---------------
    @param m: Side parallel to the x-axis
    @param n: Side parallel to the y-axis
    @param I: Current in the contour
    @param spacing: Spacing between coil and the calculation domain boundary
    @param cp: Calculation domain points
    @return: Z-component B of the field of single coil
    """
    mu0 = np.pi * 4e-7
    calc_radius = max_side * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid

    xv, yv, zv = transposition(xv, yv, zv)

    C = mu0 * I / (4 * np.pi)

    c1 = m / 2 + xv
    c2 = m / 2 - xv
    c3 = -c2
    c4 = -c1

    d1 = yv + n / 2
    d2 = yv + n / 2
    d3 = yv - n / 2
    d4 = yv - n / 2

    r1 = np.sqrt(c1 ** 2 + d1 ** 2 + zv ** 2)
    r2 = np.sqrt(c2 ** 2 + d2 ** 2 + zv ** 2)
    r3 = np.sqrt(c3 ** 2 + d3 ** 2 + zv ** 2)
    r4 = np.sqrt(c4 ** 2 + d4 ** 2 + zv ** 2)

    Bz_square = C * ((-d1 / (r1*(r1 + c1)) - c1 / (r1*(r1 + d1))) + (d2 / (r2*(r2 - c2)) - c2 / (r2*(r2 + d2))) +
                     (-d3 / (r3*(r3 + c3)) - c3 / (r3*(r3+d3))) + (d4 / (r4*(r4-c4)) - c4 / (r4*(r4+d4))))

    return Bz_square


def Bz_square_contour(R, X_side, Y_side, I, spacing, cp):
    """
    Calculates the Bz field for a square contour
    ---------------
    """
    I = np.sqrt(2) * I
    max_side = max([X_side, Y_side])
    X_sides, Y_sides = Radii_in_sides_square(R, X_side, Y_side)

    Bz_square_contour = np.zeros([cp, cp, cp])


    for x, y in zip(X_sides, Y_sides):
        Bz_square_contour += Bz_square_single(x, y, I, spacing, cp, max_side)

    return Bz_square_contour
import numpy as np
import math
from scipy.special import ellipk,ellipkm1, ellipe


def prop_coeff(R):
    """
    Calculates the radius reduction factor
    ---------------
    """
    R.sort()
    prop = []
    for i in range(len(R)-1):
        prop.append(R[i]/R[i+1])

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


def Bz_segment(x1, y1, x2, y2, g, I, spacing, cp):
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
    mu0 = np.pi * 4e-7
    C = mu0 * I / (4 * np.pi)

    calc_radius = g * spacing
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)

    if x1 != x2 and y1 != y2:
        k = (y2 - y1) / (x2 - x1)
        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        alpha = yv - (xv * k + b)
        betta = k ** 2 + 1
        gamma = k * (yv - b) + xv

        Bz_segment1 = (alpha * (betta * x1 - gamma)) / ((betta ** 2 * zv ** 2 + alpha ** 2) * np.sqrt(
            betta * x1 ** 2 - 2 * gamma * x1 + zv ** 2 + (yv - b) ** 2 + xv ** 2))
        Bz_segment2 = (alpha * (betta * x2 - gamma)) / ((betta ** 2 * zv ** 2 + alpha ** 2) * np.sqrt(
            betta * x2 ** 2 - 2 * gamma * x2 + zv ** 2 + (yv - b) ** 2 + xv ** 2))

        Bz_segment = C * (Bz_segment2 - Bz_segment1)
    elif x1 == x2 and y1 != y2:
        x = x1
        alpha = zv**2 + (xv - x)**2

        Bz_segment1 = ((x - xv) * (y1 - yv)) / (alpha * np.sqrt((y1 - yv)**2 + alpha))
        Bz_segment2 = ((x - xv) * (y2 - yv)) / (alpha * np.sqrt((y2 - yv) ** 2 + alpha))

        Bz_segment = C * (Bz_segment2 - Bz_segment1)

    elif x1 != x2 and y1 == y2:
        y = y1
        alpha = zv ** 2 + (yv - y) ** 2

        Bz_segment1 = ((yv - y) * (x1 - xv)) / (alpha * np.sqrt((x1 - xv) ** 2 + alpha))
        Bz_segment2 = ((yv - y) * (x2 - xv)) / (alpha * np.sqrt((x2 - xv) ** 2 + alpha))

        Bz_segment = C * (Bz_segment2 - Bz_segment1)

    return Bz_segment


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
    if not direction:
        I = -I

    I = np.sqrt(2)*I

    Bz_piecewise_linear_contour_single = np.zeros((cp, cp, cp))
    for i in range(len(coords)):
        try:
            Bz_piecewise_linear_contour_single += Bz_segment(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], g, I, spacing, cp)
        except IndexError:
            Bz_piecewise_linear_contour_single += Bz_segment(coords[0][0], coords[0][1], coords[i][0], coords[i][1], g, I, spacing, cp)

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

    g = np.amax(l)

    Bz_piecewise_linear_contour = np.zeros([cp, cp, cp])

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
    Bz_circular_contour = np.zeros([cp, cp, cp])

    for r in R:
        Bz_circular_contour += Bz_circular_single(max(R), r, I, spacing, cp)


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

    C = mu0 * I / (4 * np.pi)

    c1 = xv + m / 2
    c2 = -xv + m / 2
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

    Bz_square = C * (-c2 / (r2 * (r2 + d2)) + d2 / (r2 * (r2 - c2)) - c1 / (r1 * (r1 + d1)) - d1 / (r1 * (r1 + c1)) -
                     c4 / (r4 * (r4 + d4)) + d4 / (r4 * (r4 - c4)) - c3 / (r3 * (r3 + d3)) - d3 / (r3 * (r3 + c3)))

    return Bz_square


def Bz_square_contour(R, X_side, Y_side, I, spacing, cp):
    """
    Calculates the Bz field for a square contour
    ---------------
    """
    max_side = max([X_side, Y_side])
    X_sides, Y_sides = Radii_in_sides_square(R, X_side, Y_side)

    Bz_square_contour = np.zeros([cp, cp, cp])

    for x, y in zip(X_sides, Y_sides):
        Bz_square_contour += Bz_square_single(x, y, I, spacing, cp, max_side)

    return Bz_square_contour
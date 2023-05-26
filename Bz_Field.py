import numpy as np
import math
from scipy.special import ellipk,ellipkm1, ellipe
from COV import calculation_plane


def prop_coeff(R):
    """
    Calculates the radius reduction factor
    ---------------
    """
    R.sort()
    R.reverse()
    prop = []
    for i in range(len(R)):
        prop.append(R[i]/max(R))

    return prop


def Radii_in_sides_square(R, X_side, Y_side, split = False):
    """
    Converts radii to sides of a rectangular coil
    ---------------
    """
    if split:
        split_sides = []
        for coil in R:
            prop = prop_coeff(coil)
            sides = []
            for k in prop:
                sides.append([X_side*k, Y_side*k])
            split_sides.append(sides)
        return split_sides
    else:
        prop = prop_coeff(R=R)
        X_sides, Y_sides = [], []
        for k in prop:
            X_sides.append(X_side * k)
            Y_sides.append(Y_side * k)

        return X_sides, Y_sides


def Radii_in_coords(R, coords_max, split=False):
    """
    Converts radii to vertex coordinates of a piecewise linear coil
    ---------------
    """
    if split:
        split_list_of_coords = []
        for coil in R:
            prop = prop_coeff(R=coil)
            list_of_coords = []
            for k in prop:
                new_coords = []
                for point in coords_max:
                    new_coords.append([point[0] * k, point[1] * k])
                list_of_coords.append(new_coords)
            split_list_of_coords.append(list_of_coords)
        return split_list_of_coords
    else:
        prop = prop_coeff(R=R)
        list_of_coords = []
        for k in prop:
            new_coords = []
            for point in coords_max:
                new_coords.append([point[0] * k, point[1] * k])
            list_of_coords.append(new_coords)
        return list_of_coords


def Bz_segment(start_point, end_point, I, cp, calc_radius, view_plane):
    """
    Calculates z-component of B field of single-segment
    ---------------
    @param start_point: The beginning of the segment
    @param end_point: The end of the segment
    @param I: Current in the contour
    @param cp: Calculation domain points
    @param calc_radius: Calculation domain length
    @param view_plane: Calculation plane
    @return: Z-component of B field of single-segment
    """
    x1, y1 = start_point[0], start_point[1]
    x2, y2 = end_point[0], end_point[1]
    mu0 = np.pi * 4e-7
    C = mu0 * I / (4 * np.pi)

    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)

    if x1 != x2 and y1 != y2:

        k = (y2 - y1) / (x2 - x1)
        b = y2 - k*x2

        alpha = np.sqrt(k**2 + 1)
        betta = (xv[:, :, view_plane] + k*yv[:, :, view_plane] - b*k) / (alpha**2)
        gamma = yv[:, :, view_plane] - (k*xv[:, :, view_plane] + b)
        delta = zv[:, :, view_plane]**2 + (gamma / alpha)**2

        Bz_segment_1 = C * ((x1 - betta) * gamma) / (delta * np.sqrt((alpha * (x1 - betta))**2 + delta))
        Bz_segment_2 = C * ((x2 - betta) * gamma) / (delta * np.sqrt((alpha * (x2 - betta))**2 + delta))

        return Bz_segment_2 - Bz_segment_1

    elif x1 == x2 and y1 != y2:

        alpha = zv[:, :, view_plane]**2 + (x1 - xv[:, :, view_plane])**2

        Bz_segment_1 = C * ((x1 - xv[:, :, view_plane]) * (y1 - yv[:, :, view_plane])) / (alpha * np.sqrt((y1 - yv[:, :, view_plane])**2 + alpha))
        Bz_segment_2 = C * ((x1 - xv[:, :, view_plane]) * (y2 - yv[:, :, view_plane])) / (alpha * np.sqrt((y2 - yv[:, :, view_plane])**2 + alpha))

        return Bz_segment_2 - Bz_segment_1

    elif x1 != x2 and y1 == y2:

        alpha = zv ** 2 + (y1 - yv) ** 2

        Bz_segment_1 = C * ((yv[:, :, view_plane] - y1) * (x1 - xv[:, :, view_plane])) / (alpha * np.sqrt((x1 - xv[:, :, view_plane]) ** 2 + alpha))
        Bz_segment_2 = C * ((yv[:, :, view_plane] - y1) * (x2 - xv[:, :, view_plane])) / (alpha * np.sqrt((x2 - xv[:, :, view_plane]) ** 2 + alpha))

        return Bz_segment_2 - Bz_segment_1

def Bz_piecewise_linear_contour_single(coords,  I, cp, calc_radius, view_plane):
    """
    Calculates Bz field of piecewise linear coil
    ---------------
    @param coords: Coordinates of the contour corners
    @param I: Current in the contour
    @param cp: Calculation domain points
    @param calc_radius: Calculation domain length
    @param view_plane: Calculation plane
    @return: Z-component B of the field of single coil
    """

    Bz_piecewise_linear_contour_single = np.zeros((cp, cp, cp))
    for i in range(len(coords)):
        try:
            Bz_piecewise_linear_contour_single += Bz_segment(start_point=coords[i],
                                                             end_point=coords[i + 1],
                                                             I=I,
                                                             cp=cp,
                                                             calc_radius=calc_radius,
                                                             view_plane=view_plane)
        except IndexError:
            Bz_piecewise_linear_contour_single += Bz_segment(start_point=coords[i],
                                                             end_point=coords[0],
                                                             I=I,
                                                             cp=cp,
                                                             calc_radius=calc_radius,
                                                             view_plane=view_plane)

    return Bz_piecewise_linear_contour_single


def Bz_piecewise_linear_contour(R, coords,  I, P, cp, height, direction=False):
    """
    Calculates the Bz field for a piecewise linear contour
    ---------------
    @param R: Array of radii
    @param coords: Coordinates of the largest turn
    @param I: Current in the contour
    @param P: The boundary of the calculation of the COV
    @param cp: Calculation domain points
    @param height: Height above the coil [m]
    @param direction: The direction of the current along the contour. If the current flows clockwise, then by default this value is True
    @return: Z-component B of the field of a piecewise linear contour
    """
    if not direction:
        I = -I

    list_of_coords = Radii_in_coords(R, coords)

    l = []
    for i in range(len(coords)):
        l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))

    calc_radius = np.amax(l) * P
    cell_size = 2 * calc_radius / (cp - 1)
    view_plane = calculation_plane(cell_size=cell_size,
                                   height=height,
                                   cp=cp)

    Bz_piecewise_linear_contour = np.zeros((cp, cp, cp))

    for coil in list_of_coords:
        Bz_piecewise_linear_contour += Bz_piecewise_linear_contour_single(coords=coil,
                                                                          I=I,
                                                                          cp=cp,
                                                                          calc_radius=calc_radius,
                                                                          view_plane=view_plane)

    return Bz_piecewise_linear_contour


def Bz_circular_single(a, I, cp, calc_radius, view_plane):
    """
    Calculates the Bz field of a single circular coil
    ---------------
    @param a: Turn radius
    @param I: Current in the contour
    @param cp: Calculation domain points
    @param calc_radius: Calculation domain length
    @param view_plane: Calculation plane
    @return: Z-component B of the field of a single circular coil
    """
    mu0 = np.pi * 4e-7
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid

    ro = np.sqrt(xv[:, :, view_plane] ** 2 + yv[:, :, view_plane] ** 2)
    r = np.sqrt(xv[:, :, view_plane] ** 2 + yv[:, :, view_plane] ** 2 + zv[:, :, view_plane] ** 2)
    C = mu0 * I / np.pi
    alpha = np.sqrt(a ** 2 + r ** 2 - 2 * a * ro)
    beta = np.sqrt(a ** 2 + r ** 2 + 2 * a * ro)
    k = np.sqrt(1 - alpha ** 2 / beta ** 2)

    Bz = C / (2 * alpha ** 2 * beta) * ((a ** 2 - r ** 2) * ellipe(k ** 2) + alpha ** 2 * ellipk(k ** 2))

    return Bz


def Bz_circular_contour(R, I, P, cp, height):
    """
    Calculates the Bz field for a circular contour
    ---------------
    @param R: Array of radii
    @param I: Current in the contour
    @param P: The boundary of the calculation of the COV
    @param cp: Calculation domain points
    @param height: Height above the coil [m]
    @return: Z-component B of the field of a circular contour
    """
    Bz_circular_contour = np.zeros((cp, cp, cp))

    calc_radius = max(R) * P
    cell_size = 2 * calc_radius / (cp -1)
    view_plane = calculation_plane(cell_size=cell_size,
                                   height=height,
                                   cp=cp)

    for r in R:
        Bz_circular_contour += Bz_circular_single(a=r,
                                                  I=I,
                                                  cp=cp,
                                                  calc_radius=calc_radius,
                                                  view_plane=view_plane)
    return Bz_circular_contour


def Bz_rectangle_single(m, n, I, cp, calc_radius, view_plane):
    """
    Calculates the Bz field of a single square coil
    ---------------
    @param m: Side parallel to the x-axis
    @param n: Side parallel to the y-axis
    @param I: Current in the contour
    @param cp: Calculation domain points
    @param calc_radius: Calculation domain length
    @param view_plane: Calculation plane
    @return: Z-component B of the field of single coil
    """
    mu0 = np.pi * 4e-7
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid

    C = mu0 * I / (4 * np.pi)

    c1 = xv[:, :, view_plane] + m / 2
    c2 = -xv[:, :, view_plane] + m / 2
    c3 = -c2
    c4 = -c1

    d1 = yv[:, :, view_plane] + n / 2
    d2 = yv[:, :, view_plane] + n / 2
    d3 = yv[:, :, view_plane] - n / 2
    d4 = yv[:, :, view_plane] - n / 2

    r1 = np.sqrt(c1 ** 2 + d1 ** 2 + zv[:, :, view_plane] ** 2)
    r2 = np.sqrt(c2 ** 2 + d2 ** 2 + zv[:, :, view_plane] ** 2)
    r3 = np.sqrt(c3 ** 2 + d3 ** 2 + zv[:, :, view_plane] ** 2)
    r4 = np.sqrt(c4 ** 2 + d4 ** 2 + zv[:, :, view_plane] ** 2)

    Bz_rectangle = C * (-c2 / (r2 * (r2 + d2)) + d2 / (r2 * (r2 - c2)) - c1 / (r1 * (r1 + d1)) - d1 / (r1 * (r1 + c1)) -
                     c4 / (r4 * (r4 + d4)) + d4 / (r4 * (r4 - c4)) - c3 / (r3 * (r3 + d3)) - d3 / (r3 * (r3 + c3)))

    return Bz_rectangle


def Bz_rectangle_contour(R, X_side, Y_side, I, P, cp, height):
    """
    Calculates the Bz field for a square contour
    ---------------
    @param R: Array of radii
    @param X_side: The largest side parallel to the x-axis
    @param Y_side: The largest side parallel to the y-axis
    @param I: Current in the contour
    @param P: The boundary of the calculation of the COV
    @param cp: Calculation domain points
    @param height: Height above the coil [m]
    @return: Z-component B of the field of a rectangle contour
    """
    X_sides, Y_sides = Radii_in_sides_square(R, X_side, Y_side)

    Bz_rectangle_contour = np.zeros((cp, cp, cp))

    calc_radius = 0.5 * max([X_side, Y_side]) * P
    cell_size = 2 * calc_radius / (cp - 1)
    view_plane = calculation_plane(cell_size=cell_size,
                                   height=height,
                                   cp=cp)

    for x, y in zip(X_sides, Y_sides):
        Bz_rectangle_contour += Bz_square_single(m=x,
                                              n=y,
                                              I=I,
                                              cp=cp,
                                              calc_radius=calc_radius,
                                              view_plane=view_plane)

    return Bz_rectangle_contour
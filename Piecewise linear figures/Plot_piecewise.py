import matplotlib.pyplot as plt
import numpy as np
from Bz_Field_piecewise import Radii_in_coords, transposition


def plot_2d(Bz, height, a_max, spacing, cp):
    """


    """
    calc_radius = a_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    view_plane = int(height / (2 * calc_radius / cp) + cp / 2)
    fig = plt.figure()
    plt.plot(x * 1e2, Bz[:, view_plane, view_plane] * 1e6)
    plt.xlabel('x [cm]')
    plt.ylabel('Bz [uT]')
    plt.title('Bz Field at {} mm height'.format(height * 1e3))

    return fig


def plot_3d(Bz, height, a_max, spacing, cp):
    """


    """
    calc_radius = a_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid
    xv, yv, zv = transposition(xv, yv, zv)
    view_line = height / (2 * calc_radius / np.size(x)) + np.size(x) / 2
    view_line = int(view_line)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv[:, :, 1] * 1e2, yv[:, :, 1] * 1e2, Bz[:, :, view_line] * 1e6, cmap='inferno')
    ax.set_title('Bz Field at {} mm height'.format(height * 1e3))
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')

    return fig


def plot_piecewise_linear_coil(coords_max, spacing, R):
    """
    Draws a piecewise linear coil
    ---------------
    @param coords_max: Coordinates of the largest turn
    @param spacing: Spacing between coil and the calculation domain boundary
    @param R: Set of radii
    """
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.subplots()
    l = []
    for i in range(len(coords_max)-1):
        l.append(np.sqrt(coords_max[i][0]**2 + coords_max[i][1]**2))

    r_max = max(l) * spacing

    ax.set_xlim((-r_max, r_max))
    ax.set_ylim((-r_max, r_max))

    list_of_coords = Radii_in_coords(R, coords_max)

    for coords in list_of_coords:
        for i in range(len(coords)):
            try:
                plt.plot([coords[i][0], coords[i + 1][0]], [coords[i][1], coords[i + 1][1]], color='#000000')
            except IndexError:
                plt.plot([coords[i][0], coords[0][0]], [coords[i][1], coords[0][1]], color='#000000')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    return fig
import matplotlib.pyplot as plt
import numpy as np
from Bz_field import Radii_in_coords, Radii_in_sides_square


def plot_2d(Bz, height, a_max, spacing, cp):
    """


    """
    calc_radius = a_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    view_line = height / (2 * calc_radius / np.size(x)) + np.size(x) / 2
    view_line = int(view_line)
    fig = plt.figure()
    plt.plot(x * 1e2, Bz[:, view_line, view_line] * 1e6)
    plt.xlabel('x [cm]')
    plt.ylabel('Bz [uT]')
    plt.title('Bz Field at {} mm height'.format(height * 1e3))

    plt.show()


def plot_3d(Bz, height, a_max, spacing, cp):
    """


    """
    calc_radius = a_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid
    view_line = height / (2 * calc_radius / np.size(x)) + np.size(x) / 2
    view_line = int(view_line)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv[:, :, 1] * 1e2, yv[:, :, 1] * 1e2, Bz[:, :, view_line] * 1e6, cmap='inferno')
    ax.set_title('Bz Field at {} mm height'.format(height * 1e3))
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')


def plot_vector(Bz, height, a_max, spacing, cp):
    """


    """
    calc_radius = a_max * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    xv, yv, zv = np.meshgrid(x, x, x)  # Creating meshgrid
    view_line = height / (2 * calc_radius / np.size(x)) + np.size(x) / 2
    view_line = int(view_line)
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.quiver(xv[::30, ::30, ::30], yv[::30, ::30, ::30], zv[::30, ::30, ::30], Bx_sum[::30, ::30, ::30],
              By_sum[::30, ::30, ::30], Bz_sum[::30, ::30, ::30], length=0.03, normalize=True)

    ax.set_title('Vector field'.format(height * 1e3))
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.show()


def plot_coil(a_max, spacing, R):
    """
    Draws a circular coil
    ---------------
    @param R: Radius of the largest turn
    @param spacing: Spacing between coil and the calculation domain boundary
    @param R: Set of radii
    """
    fig = plt.figure(figsize=(3, 3), dpi=300)

    ax = fig.subplots()
    ax.set_xlim((-a_max * spacing, a_max * spacing))
    ax.set_ylim((-a_max * spacing, a_max * spacing))
    for i, radius in enumerate(R):
        # circle = plt.Circle((a_max*spacing, a_max*spacing), radius, fill=False)
        circle = plt.Circle((0, 0), radius, fill=False)

        ax.add_patch(circle)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    plt.show()


def plot_square_coil(m_max, n_max, spacing, R):
    """
    Draws a rectangular coil
    ---------------
    @param m: Side parallel to the x-axis of the largest turn
    @param n: Side parallel to the y-axis of the largest turn
    @param spacing: Spacing between coil and the calculation domain boundary
    @param R: Set of radii
    """
    m_i, n_i = R_in_sides_square(R, m_max, n_max)

    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.subplots()
    max_size = np.max([m_max, n_max]) * spacing

    ax.set_xlim((-max_size, max_size))
    ax.set_ylim((-max_size, max_size))

    for i, size_m in enumerate(m_i):
        for j, size_n in enumerate(n_i):
            if i == j:
                rec = plt.Rectangle((-size_m / 2, -size_n / 2), size_m, size_n, fill=False)
                ax.add_patch(rec)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    plt.show()


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
                plt.plot([coords[0][0], coords[i][0]], [coords[0][1], coords[i][1]], color='#000000')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    plt.show()
import numpy as np
import math


def dist(x1, y1, x2, y2):
    """
    Return the distance between two points A(x1,y1) and B(x2,y2)

    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Cirqular mask for COV calculation
def mask_circle(tiles, cx, cy, r):
    """
    Creates circular binary mask in array "tiles" with center in point C(cx,cy)
    and radius "r"

    """
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x][y] = 1


# COV for cirqular coil
def COV_circ(Bz, max_coil_r, height, spacing):
    """
    Coefficient of variation calculation for circular coil

    Parameters
    ----------
    Bz :
        Field for CoV calculation
    max_coil_r :
        Maximum coil radius [m]
    height :
        Height above the coil
    spacing :
        Spacing between coil and the calculation domain boundary

    """
    calc_radius = max_coil_r * spacing  # Calculation domain length

    view_line = height / (2 * calc_radius / len(Bz)) + len(Bz) / 2
    view_line = int(view_line)

    cp = len(Bz)  # Calculation domain
    cx = cp // 2  # center of calc area
    cy = cp // 2
    cell_size = calc_radius / cp
    tiles = np.zeros([cp, cp])
    r_cov_m = max_coil_r * 0.9  # Uniform area
    r_cov = r_cov_m / cell_size / 2  # Uniform area in cells
    mask_circle(tiles, cx, cy, round(r_cov))
    Bz_masked = np.multiply(Bz[:, :, view_line], tiles)
    B_mean = np.sum(Bz_masked) / np.sum(tiles)
    B_std = np.sqrt(np.sum((Bz_masked - np.multiply(B_mean, tiles)) ** 2) / np.sum(tiles))
    COV = B_std / B_mean
    return COV

"""COV for square coil"""
def COV_square(Bz, max_side, height, spacing):
    """
    Coefficient of variation calculation for square coil
     ---------------
     @param Bz: Field for CoV calculation
     @param max_side: Maximum side of square
     @param height:
    """
    calc_radius = max_side * spacing

    view_line = int(height / (2 * calc_radius / len(Bz)) + len(Bz) / 2)
import numpy as np
import math


def dist(x1, y1, x2, y2):
    """
    Calculates the distance between two points
    ---------------
    @param x1, y1: Сoordinates of the first point
    @param x2, y2: Coordinates of the second point
    @return: The distance  between two points
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def mask_circular(tiles, cx, cy, r):
    """
    Creates a circular binary mask
    ---------------
    @param tiles: Zero two-dimensional array
    @param cx, cy: Center of the calculation domain
    @param r: Radius of the calculation domain is in points
    @return: A circular binary mask
    """
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x][y] = 1

def mask_square(tiles, cx, cy, max_side, min_side):
    """
    Creates a square binary mask
    ---------------
    @param tiles: Zero two-dimensional array
    @param cx, cy: Center of the calculation domain
    @param max_side: The largest side of the calculation domain is in points
    @param min_side: The smallest side of the calculation domain is in points
    @return: A square binary mask
    """
    for x in range(cx-max_side, cx+max_side):
        for y in range(cy-min_side, cy+min_side):
            tiles[x][y] = 1

def mask_piecewise_linear(tiles, coords):
    """
    Creates a piecewise linear binary mask
    --------------
    @param tiles: Zero two-dimensional array
    @param coords: Coordinate indexes
    @return: A piecewise linear mask
    """
    for i in range(len(coords)):
        try:
            x1, y1 = coords[i][0], coords[i][1]
            x2, y2 = coords[i + 1][0], coords[i + 1][1]
            X = []
            for x in range(x1, x2 + 1):
                X.append(x)
            Y = []
            for y in range(y1, y2 + 1):
                Y.append(y)
            for x, y in zip(X, Y):
                tiles[x][y] = 1
        except IndexError:
            x1, y1 = coords[0][0], coords[0][1]
            x2, y2 = coords[i][0], coords[i][1]
            X = []
            for x in range(x1, x2 + 1):
                X.append(x)
            Y = []
            for y in range(y1, y2 + 1):
                Y.append(y)
            for x, y in zip(X, Y):
                tiles[x][y] = 1
    for y in range(len(tiles)):
        cut = tiles[:, y]
        indexes = []
        for x in len(cut):
            if cut[x]==1:
                indexes.append(x)
        for x in range(indexes[0], indexes[1]+1):
            tiles[x][y] = 1


def COV_circle(Bz, max_coil_r, height, spacing, P):
    """
    Calculates the coefficient of variation for a circular coil
    --------------
    @param Bz: Field for calculating the COV
    @param max_coil_r: The largest radius of a circular coil [m]
    @param height: Height above the coil [m]
    @param spacing: Spacing between coil and the calculation domain boundary
    @param P: The boundary of the calculation of the COV
    @return: COV
    """
    calc_radius = max_coil_r * spacing  # Calculation domain length

    view_line = height / (2 * calc_radius / len(Bz)) + len(Bz) / 2
    view_plane  =len(Bz)/2 * (height / calc_radius + 1)
    view_line = int(view_line)

    cp = len(Bz)  # Calculation domain
    cx = cp // 2  # center of calc area
    cy = cp // 2
    cell_size = 2*calc_radius / cp
    tiles = np.zeros([cp, cp])
    r_cov_m = max_coil_r * P  # Uniform area
    r_cov = r_cov_m / cell_size  # Uniform area in cells
    mask_circle(tiles, cx, cy, round(r_cov))
    Bz_masked = np.multiply(Bz[:, :, view_line], tiles)
    B_mean = np.sum(Bz_masked) / np.sum(tiles)
    B_std = np.sqrt(np.sum((Bz_masked - np.multiply(B_mean, tiles)) ** 2) / np.sum(tiles))
    COV = B_std / B_mean
    return COV


def COV_square(Bz, max_side, min_side, height, spacing, P):
    """
    Calculates the coefficient of variation for a square coil
    ---------------
    @param Bz: Field for calculating the COV
    @param max_side: The largest side of a square [m]
    @param min_side: The smallest side of a square [m]
    @param height: Height above the coil [m]
    @param spacing: Spacing between coil and the calculation domain boundary
    @param P: The boundary of the calculation of the COV
    @return: COV
    """
    cp = len(Bz)
    cx = cp // 2
    cy = cp // 2

    calc_radius = max_side * spacing
    cell_size = 2*calc_radius / (cp+1)
    view_plane = round(height/cell_size) + 1 + round(cp / 2)
    tiles = np.zeros((0, 0))

    max_side_COV = max_side * P / cell_size
    min_side_COV = min_side * P / cell_size

    mask_square(tiles, cx, cy, round(max_side_COV), round(min_side_COV))

    Bz_masked = np.multiply(Bz[:, :, view_plane], tiles)

    Bz_mean = np.sum(Bz_masked) / np.sum(tiles)
    Bz_std = np.sqrt(np.sum((np.multiply(Bz_mean, tiles) - Bz_masked)**2) / (np.sum(tiles)))

    COV = Bz_std / Bz_mean

    return COV

def COV_piecewise_linear(Bz, coords, height, spacing, P):
    """
    Calculates the coefficient of variation for a square coil
    ---------------
    @param Bz: Field for calculating the COV
    @param coords: Coordinates of the vertices of a piecewise linear coil [m]
    @param height: Height above the coil [m]
    @param spacing: Spacing between coil and the calculation domain boundary
    @param P: The boundary of the calculation of the COV
    @return: COV
    """
    cp = len(Bz)

    l = []
    for i in range(len(coords)):
        l.append((coords[i][0])**2 + (coords[i][1])**2)

    calc_radius = max(l) * spacing
    cell_size = 2 * calc_radius / (cp + 1)
    view_plane = round(height / cell_size) + 1 + round(cp / 2)
    tiles = np.zeros((0, 0))

    coords_COV = []
    for i in coords:
        coords_COV.append([round(i[0] * P / cell_size), round(i[1] * P / cell_size)])

    mask_piecewise_linear(tiles, coords_COV)
    Bz_masked = np.multiply(tiles, Bz[:, :, view_plane])
    Bz_mean  = np.sum(Bz_masked) / np.sum(tiles)
    Bz_std = np.sqrt((np.sum((Bz_masked  -np.multiply(Bz_mean, tiles))**2)) / (np.sum(tiles)))
    COV = Bz_std / Bz_mean
    return COV











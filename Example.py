import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import Plot
import Resistance
import COV


R = [0.5, 0.6]
# coords = [[-3, -4], [-3, -3], [-4, -3], [-4, 3], [-3, 3], [-3, 4], [3, 4], [3, 3], [4, 3], [4, -3], [3, -3], [3, -4]]
# coords = [[-0.5, -np.sqrt(3)/2], [-1, 0], [-0.5, np.sqrt(3)/2], [0.5, np.sqrt(3)/2], [1, 0], [0.5, -np.sqrt(3)/2]]
# coords = [[-0.5, -np.sqrt(3)/6], [0, np.sqrt(3)/3], [0.5, -np.sqrt(3)/6]]
# coords = [[0, -np.sqrt(3)/2], [-0.5, 0], [0, np.sqrt(3)/2], [0.5, 0]]
# coords = [[0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05]]
coords = [[0.05, -0.15], [-0.05, -0.15], [-0.05, 0.15], [0.05, 0.15]]
# coords = [[-1, 7], [-1, 0], [-3, 0], [-3, -3], [0, -3], [0, -2], [1, -2], [1, -3], [4, -3], [4, 0], [2, 0], [2, 7]]
# coords = [[-1, 7], [-1, 0], [-3, 0], [-3, -3], [4, -3], [4, 0], [2, 0], [2, 7]]

spacing = 1.5
l = []
for i in range(len(coords)):
    l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))

g = np.amax(l)
I = 1
P = 0.9
spacing = 1.5
cp = 50
height = 0.03
material, l, d, nu = 'Copper', 0.4, 0.125e-3, 1e9
X_side = 0.1
Y_side = 0.3
max_side = max([X_side, Y_side])
# Bz = Bz_Field.Bz_piecewise_linear_contour_single(coords, I, spacing, cp, g, False)
Bz = Bz_Field.Bz_square_single(X_side, Y_side, I, spacing, cp, max_side)
# for i in range(len(coords)):
#     coords[i][0] = round(coords[i][0]*10 + 15)
#     coords[i][1] = round(coords[i][1]*10 + 15)
# tiles = np.zeros([30, 30])
# COV = COV.COV_piecewise_linear(Bz, coords, height, spacing, P)
# print(COV)
# COV.mask_piecewise_linear(tiles, coords)
Plot.plot_2d(Bz, height, g, spacing, cp)
plt.show()
Plot.plot_3d(Bz, height, g, spacing, cp)
Plot.plot_piecewise_linear_coil(coords, spacing, R)

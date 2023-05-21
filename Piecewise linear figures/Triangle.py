import matplotlib.pyplot as plt
import numpy as np
import Bz_Field_piecewise
import COV_piecewise
import Plot_piecewise

R = [1]
coords = [[-1.5, -np.sqrt(3)/2], [0, np.sqrt(3)], [1.5, -np.sqrt(3)/2]]
I = 1
P = 0.9
spacing = 1.5
cp = 50
height = 0.015

l = []
for i in range(len(coords)):
    l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))

g = max(l)

Bz_piecewise = Bz_Field_piecewise.Bz_piecewise_linear_contour_single(coords, I, spacing, cp, False)
Plot_piecewise.plot_3d(Bz_piecewise, height, g, spacing, cp)
plt.show()
Plot_piecewise.plot_2d(Bz_piecewise, height, g, spacing, cp)
plt.show()
Plot_piecewise.plot_piecewise_linear_coil(coords, spacing, R)
plt.show()
COV = COV_piecewise.COV_piecewise_linear(Bz_piecewise, coords, height, spacing, P)
print(round(COV*100, 1), '%')
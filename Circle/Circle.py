import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import COV
import Plot
import Resistance
from turns_splitter import split


I = 1
freq = 6.78e6
material = 'Copper'
D = 0.002
P = 0.9
spacing = 1.5
cp = 90
height = 0.08
split_R = split(R, freq)
print(R, '\n', split_R)
l = []
for i in range(len(coords)):
    l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))

g = max(l)

Bz_piecewise = Bz_Field.Bz_piecewise_linear_contour(R, coords, I, spacing, cp)
Plot.plot_3d(Bz_piecewise, height, g, spacing, cp)
plt.show()
Plot.plot_2d(Bz_piecewise, height, g, spacing, cp)
plt.show()
Plot.plot_piecewise_linear_coil(coords, spacing, R)
plt.show()
COV = COV.COV_circle(Bz_piecewise, coords, height, spacing, P)
print(round(COV*100, 1), '%')

# lengths = Resistance.length_piecewise_linear_coils(Bz_Field.Radii_in_coords(split_R, coords, split=True))
# length = sum(lengths)
# Resistance = Resistance.resistance_contour(lengths, material, D, freq)
# print(length, Resistance)



import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import COV
import Plot
import Resistance
from turns_splitter import split


# R = [1]
# R = [0.9947214076246335, 0.9883977360359807, 0.9831564514575973, 0.9779397780543757, 0.97359784047745, 0.969272112284538, 0.9545826233095508, 0.9129407127854929, 0.46823283225180146, 0.10038421555263972]
# R = [0.9912023460410557, 0.9814188841624502, 0.9814188841624502, 0.9779384542151157, 0.9744675518320953, 0.9744675518320953, 0.9744675518320953, 0.9640675413048553, 0.9494969335238549, 0.5209116065925705, 0.1173747707341395, 0.10140225587788611, 0.1]
R = [0.9648093841642229, 0.9299546805496933, 0.9291103726345868, 0.9282658666533861, 0.9282658666533861, 0.4964205268096654, 0.13910411814157464, 0.1]

coords = [[0, -2], [-0.4142, -1], [-1.4142, -1.4142], [-1, -0.4142], [-2, 0], [-1, 0.4142], [-1.4142, 1.4142], [-0.4142, 1], [0, 2], [0.4142, 1], [1.4142, 1.4142], [1, 0.4142], [2, 0], [1, -0.4142], [1.4142, -1.4142], [0.4142, -1]]
I = 1
freq = 6.78e6
material = 'Copper'
D = 0.002
P = 0.7
spacing = 1.5
cp = 200
height = 0.03
# split_R = split(R, freq)
# print(R, '\n', split_R)
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
COV = COV.COV_piecewise_linear(Bz_piecewise, coords, height, spacing, P)
print(round(COV*100, 1), '%')

# lengths = Resistance.length_piecewise_linear_coils(Bz_Field.Radii_in_coords(split_R, coords, split=True))
# length = sum(lengths)
# Resistance = Resistance.resistance_contour(lengths, material, D, freq)
# print(length, Resistance)
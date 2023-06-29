import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import COV
import Plot
import Resistance
from turns_splitter import split

# R = [0.488, 0.485, 0.0670072840790843, 0.491, 0.482, 0.1786316337148804, 0.494, 0.4513225806451613, 0.2517835587929241, 0.497, 0.42135691987513, 0.3055369406867846, 0.5, 0.3931331945889698, 0.35355879292403747]
R = np.linspace(0.01,0.1,10)
I = 1
freq = 6.78e6
material = 'Copper'
D = 0.002
P = 0.9
spacing = 1.5
cp = 100
height = 0.015
a_max = 0.1
# split_R = split(R, freq)
# print(R, '\n', split_R)

Bz_circle = Bz_Field.Bz_circular_contour(R, I, spacing, cp)
# Plot.plot_3d(Bz_circle, height, a_max, spacing, cp)
# plt.show()
Plot.plot_2d(Bz_circle, height, a_max, spacing, cp)
# plt.show()
plt.savefig('Circular_2D_E.png')
Plot.plot_coil(a_max, spacing, R)
# plt.show()
plt.savefig('Circular_AM_E.png')
COV = COV.COV_circle(Bz_circle, a_max, height, spacing, P)
print(round(COV*100, 1), '%')

# lengths = Resistance.length_circular_coils(split_R)
# length = sum(lengths)
# Resistance = Resistance.resistance_contour(lengths, material, D, freq)
# print(length, Resistance)

# if a == 'ok':
#     plt.savefig('graph(C_0).png')


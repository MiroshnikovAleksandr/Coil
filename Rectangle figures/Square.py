import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import COV
import Plot

R = [1]
X_side = 2
Y_side = 2
I = 1
P = 0.9
spacing = 1.5
cp = 50
height = 0.015

g = 0.5 * np.sqrt(X_side**2 + Y_side**2)

Bz_square = Bz_Field.Bz_square_contour(R, X_side, Y_side, I, spacing, cp)

# Plot.plot_3d(Bz_square, height, g, spacing, cp)
# plt.show()
Plot.plot_2d(Bz_square, height, g, spacing, cp)
plt.show()
# Plot.plot_square_coil(X_side, Y_side, spacing, R)
# plt.show()

COV = COV.COV_square(Bz_square, X_side, Y_side, height, spacing, P)
print(round(COV*100, 1), '%')
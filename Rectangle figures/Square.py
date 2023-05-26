import matplotlib.pyplot as plt
import numpy as np
import Bz_Field
import COV
import Plot

# R = [1.0, 0.99, 0.98, 0.8016129032258065, 0.5645785639958377, 0.1]
# R = [1.0, 0.99, 0.98, 0.7735483870967742, 0.4601664932362123, 0.14292403746097815]
# R = [1.0, 0.99, 0.98, 0.8016129032258065, 0.6530697190426639, 0.16540062434963582]
R = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9129032258064516, 0.8492924037460977, 0.7881997918834548, 0.7296253902185225, 0.6515088449531738, 0.5370239334027055, 0.4513839750260147, 0.30115504682622274, 0.1]
X_side = 1
Y_side = 1
I = 1
P = 0.9
spacing = 1.5
cp = 50
height = 0.015

g = 0.5 * np.sqrt(X_side**2 + Y_side**2)

Bz_square = Bz_Field.Bz_square_contour(R, X_side, Y_side, I, spacing, cp)

# Plot.plot_3d(Bz_square, height, g, spacing, cp)
# plt.show()
# Plot.plot_2d(Bz_square, height, g, spacing, cp)
# plt.show()
# Plot.plot_square_coil(X_side, Y_side, spacing, R)
# plt.show()

COV = COV.COV_square(Bz_square, X_side, Y_side, height, spacing, P)
print(round(COV*100, 1), '%')
import COV, Bz_Field, Resistance, Plot, Field_functions
import see_on_arrays as np
import matplotlib.pyplot as plt


Array_radii = [1, 0.6, 0.4, 0.2]
X_side = 0.1
Y_side = 0.1
cp = 50
spacing = 1.5
height = 0.015
P = 0.9
I = 1
max_side = max([X_side, Y_side])

Bz_Nikita = Bz_Field.Bz_square_single(X_side, Y_side, I, spacing, cp, max_side)
Bz_Nikita_Array = Bz_Field.Bz_square_contour(Array_radii, X_side, Y_side, I, spacing, cp)
Plot.plot_2d(Bz_Nikita_Array, height, max_side, spacing, cp)
# Plot.plot_2d(Bz_Nikita, height, max_side, spacing, cp)
plt.show()
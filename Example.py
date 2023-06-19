# import matplotlib.pyplot as plt
# import numpy as np
# import Bz_Field
# import Plot
# import Resistance
# import COV
#
#
# R = [1]
# # coords = [[-3, -4], [-3, -3], [-4, -3], [-4, 3], [-3, 3], [-3, 4], [3, 4], [3, 3], [4, 3], [4, -3], [3, -3], [3, -4]]
# coords = [[-0.5, -0.866025], [-1, 0], [-0.5, 0.866025], [0.5, 0.866025], [1, 0], [0.5, -0.866025]]
# # coords = [[-1.5, -np.sqrt(3)/2], [0, np.sqrt(3)], [1.5, -np.sqrt(3)/2]]
# # coords = [[0, -0.866025], [-0.5, 0], [0, 0.866025], [0.5, 0]]
# coords = [[0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05]]
# # coords = [[-1, 7], [-1, 0], [-3, 0], [-3, -3], [0, -3], [0, -2], [1, -2], [1, -3], [4, -3], [4, 0], [2, 0], [2, 7]]
# # coords = [[-1, 7], [-1, 0], [-3, 0], [-3, -3], [4, -3], [4, 0], [2, 0], [2, 7]]
# # coords = [[-0.05, -0.15], [-0.05, 0.15], [0.05, 0.15], [0.05, -0.15]]
#
# l = []
# for i in range(len(coords)):
#     l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))
#
# g = max(l)
# I = 1
# P = 0.9
# spacing = 1.5
# cp = 50
# height = 0.015
# X_side = 0.1
# Y_side = 0.3
# material, l, d, nu = 'Copper', 0.4, 0.125e-3, 1e9
# Bz_piecewise = Bz_Field.Bz_piecewise_linear_contour_single(coords, I, spacing, cp, False)
# # print(Bz_piecewise[16][16][16])
# # Plot.plot_2d(Bz_piecewise, height, g, spacing, cp)
# Plot.plot_3d(Bz_piecewise, height, g, spacing, cp)
# plt.show()
# #
# # Bz_square = Bz_Field.Bz_square_single(X_side, Y_side, I, spacing, cp, max([X_side, Y_side]))
# # Plot.plot_2d(Bz_square, height, max([X_side, Y_side]), spacing, cp)
# # plt.show()
#
# # for i in range(len(coords)):
# #     coords[i][0] = round(coords[i][0]*10 + 15)
# #     coords[i][1] = round(coords[i][1]*10 + 15)
# # tiles = np.zeros([30, 30])
# # COV = COV.COV_piecewise_linear(Bz, coords, height, spacing, P)
# # COV_square = COV.COV_square(Bz_square, X_side, Y_side, height, spacing, P)
# # COV_piecewise = COV.COV_piecewise_linear(Bz_piecewise, coords, height, spacing, P)
# # print(COV_square, COV_piecewise)
# # COV.mask_piecewise_linear(tiles, coords)
# # Plot.plot_2d(Bz, height, g, spacing, cp)
# # Plot.plot_3d(Bz, height, g, spacing, cp)
# # Plot.plot_piecewise_linear_coil(coords, spacing, R)
# # plt.show()
#
# # Plot.plot_square_coil(X_side, Y_side, spacing, R)
# # plt.show()
# # x = np.linspace(-5, 5, cp)
# # xv, yv, zv = np.meshgrid(x, x, x)
# #
# # xv_T, yv_T, zv_T = Bz_Field.transposition(xv, yv, zv)
# #
# # fig = plt.figure(figsize=(7, 4), dpi=100)
# # ax_3d = fig.add_subplot(projection='3d')
#
# # ax_3d.scatter(xv[:, :, 0], yv[:, :, 0], zv[:, :, 0])
# # ax_3d.scatter(xv[:, 0, :], yv[:, 0, :], zv[:, 0, :])
# # ax_3d.scatter(xv[0, :, :], yv[0, :, :], zv[0, :, :])
#
# # ax_3d.scatter(xv_T[:, :, 0], yv_T[:, :, 0], zv_T[:, :, 0])
# # ax_3d.scatter(xv_T[:, 0, :], yv_T[:, 0, :], zv_T[:, 0, :])
# # ax_3d.scatter(xv_T[0, :, :], yv_T[0, :, :], zv_T[0, :, :])
# #
# # ax_3d.set_xlabel('x')
# # ax_3d.set_ylabel('y')
# # ax_3d.set_zlabel('z')
# #
# # plt.show()
# # print(zv, '\n \n', zv_T)

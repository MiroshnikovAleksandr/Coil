# import numpy as np
# import matplotlib.pyplot as plt
#
# import Bz_Field
# import COV
#
# cp = 3
# """
# Посмотрим на команду np.linspace():
# >>> cp = 3
# >>> x = np.linspace(-5, 5, cp)
# >>> print(x)
# Output:
# [-5.  0.  5.]
#
# Создадим сетку с помощью команды np.meshgrid()
# >>> xv, yv, zv = np.meshgrid(x, x, x)
# xv, yv, zv - трёхмерные тензоры, первый индекс которых - это его измерение,
# второй строка матрицы, а
#
# """
#
# x = np.linspace(-5, 5, cp)
# y = np.linspace(-3, -1, cp)
# z = np.linspace(1, 3, cp)
# # print(x)
# coords = [[0, -0.866025], [-0.5, 0], [0, 0.866025], [0.5, 0]]
# tiles = np.zeros((cp, cp))
# xv, yv, zv = np.meshgrid(x, x, x)
# spacing = 1.5
# l = []
# for i in range(len(coords)):
#     l.append(np.sqrt((coords[i][0])**2 + (coords[i][1])**2))
# cx, cy = cp // 2, cp // 2
# g = max(l)
# I = 1
# P = 0.9
# calc_radius = max(l) * spacing
# cell_size = 2 * calc_radius / cp
# coords_COV = []
# for i in coords:
#     coords_COV.append([round(cx + i[0] * P / cell_size), round(cy + i[1] * P / cell_size)])

# COV.mask_piecewise_linear(tiles, coords_COV)
# yv[0][1] = [1, 2, 3]
# print(xv[0][0][0], yv[0][0][0], zv[0][0][0])
# print(xv, '\n \n', yv/zv, '\n \n', zv)
# tiles = np.zeros((cp, cp))
# print(tiles)
# xv, yv, zv = np.meshgrid(x, x, x)
#
# xv_T = np.zeros((cp, cp, cp))
# for i in range(len(xv)):
#     xv_T[:, :, i] += xv[:, :, i].T
#
# zv_T = np.zeros((cp, cp, cp))
# for i in range(len(zv)):
#     zv_T[:, :, i] += zv[:, :, i].T
#
# yv_T = np.zeros((cp, cp, cp))
# for i in range(len(yv)):
#     yv_T[:, :, i] += yv[:, :, i].T

# fig = plt.figure(figsize=(7, 4), dpi=200)
# ax_3d = fig.add_subplot(projection='3d')
# # ax_3d.scatter(xv_T[:, :, 0], yv_T[:, :, 0], zv_T[:, :, 0])
# # ax_3d.scatter(xv_T[:, 0, :], yv_T[:, 0, :], zv_T[:, 0, :])
# # ax_3d.scatter(xv_T[0, :, :], yv_T[0, :, :], zv_T[0, :, :])
#
#
# for i in range(len(tiles)):
#     ax_3d.scatter(tiles[i, :], tiles[:, i])
# ax_3d.set_xlabel('x')
# ax_3d.set_ylabel('y')
# ax_3d.set_zlabel('z')
#
# plt.show()
# print(xv[:, :, 0], '\n \n', yv[:, :, 0], '\n \n', zv[:, :, 0])
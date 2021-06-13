from region import Region
import matplotlib.pyplot as plt
import sc2_math
import numpy as np

reg = Region(shape=(10, 10), value_type=int)
# for x in sc2_math.points_in_circle_np(3, (9, 9)):
#     y = (reg.shape()[0]-1, reg.shape()[1]-1)
#     if y > x >= (0, 0):
#         print(x)
#         reg.region[x] = 1
# print(reg.region)
# # print(reg.offset)
# # plt.matshow(reg.region)
# # plt.show()
# reg.draw_region()
reg.region[sc2_math.points_in_square_np(1.5, (4.5, 4.5), reg.shape())] = 1
# x = tuple(sc2_math.points_in_circle_np(3, (9, 9), reg.shape()))
# reg.region[x] = 1
reg.draw_region()
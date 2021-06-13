import matplotlib.pyplot as plt
from sc2.unit import Unit
from sc2.position import Point2

from copy import copy
from typing import Union
import numpy as np
import sc2_math


class Region:
    def __init__(self, shape: tuple, value_type=int):
        self.offset = (0, 0)
        self.region = np.zeros(dtype=value_type, shape=shape)
        pass

    def find_best_position(self, size: int) -> Point2:
        pos, = sc2_math.max_sub_matrix(self.region, size=size)
        return Point2((pos[3]+self.shape()[1], pos[1]+self.shape()[0]))

    def slice_region(self, dim: float, center_position):
        new_region = copy(self)
        if isinstance(center_position, Unit):
            center_position = center_position.position_tuple

        if (
                round(center_position[0] % 1, 2) == 0.5 and dim % 0 == 0
                or round(center_position[0] % 1, 2) == 0 and dim % 2 == 1
        ):
            dim += 1
        x, y = (round(center_position[1] - dim / 2), round(center_position[0] - dim / 2))
        new_region.region = new_region.region[x:(x + dim), y:(y + dim)]
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        new_region.offset = (x, y)
        return new_region
        pass

    def add_weight(self, weight: int, position: Union[Point2, tuple], radius: int, cycle: bool = False):
        """
        :param weight:
        :param position: Point2 or tuple
        :param radius:
        :param cycle:
        """
        if cycle:
            self.region[sc2_math.points_in_circle_np(radius, position, self.shape())] += weight
        else:
            self.region[sc2_math.points_in_square_np(radius, position, self.shape())] += weight

    def set_weight(self, weight: int, position: Union[Point2, tuple], radius: int, cycle: bool = False):
        """
        :param weight:
        :param position: Point2 or tuple
        :param radius:
        :param cycle:
        """
        if cycle:
            self.region[sc2_math.points_in_circle_np(radius, position, self.shape())] = weight
        else:
            self.region[sc2_math.points_in_square_np(radius, position, self.shape())] = weight

    def shape(self) -> tuple:
        """
        :return: return shape of the region
        """
        return self.region.shape

    def draw_region(self):
        plt.matshow(self.region)
        plt.show()
        pass

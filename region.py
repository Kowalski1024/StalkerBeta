import matplotlib.pyplot as plt
from sc2.unit import Unit
from sc2.position import Point2, Point3

from copy import copy
from typing import Union
import numpy as np
import sc2_math

_debug = False


class Region:
    def __init__(self, shape: tuple, value_type=int):
        self.offset = (0, 0)
        self.region = np.zeros(dtype=value_type, shape=shape)
        pass

    def find_best_position(self, size: int) -> Point2:
        pos, = sc2_math.max_sub_matrix(self.region, size=size)
        return Point2((pos[3], pos[1]))

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
        for x in range(round(position[0] - radius), round(position[0] + radius)):
            for y in range(round(position[1] - radius), round(position[1] + radius)):
                self.region[y][x] += weight
        pass

    def set_weight(self, weight: int, position: Union[Point2, tuple], radius: int, cycle: bool = False):
        pass

    def shape(self) -> tuple:
        return self.region.shape

    def draw_region(self):
        plt.matshow(self.region)
        plt.show()
        pass

import matplotlib.pyplot as plt
from sc2.unit import Unit
from sc2.position import Point2

from copy import copy
from abc import abstractmethod
from typing import Union, Set
import numpy as np
import sc2_math


class InfluenceGrid:
    def __init__(self, shape: tuple, init_value=0, value_type=int):
        self.offset = (0, 0)
        self.grid = np.full(dtype=value_type, shape=shape, fill_value=init_value)
        pass

    def __add__(self, other):
        if isinstance(other, InfluenceGrid):
            empty = InfluenceGrid(self.shape())
            empty.grid = self.grid + other.grid
            return empty
        else:
            raise ValueError("Wrong type")

    def __sub__(self, other):
        if isinstance(other, InfluenceGrid):
            empty = InfluenceGrid(self.shape())
            empty.grid = self.grid - other.grid
            return empty
        else:
            raise ValueError("Wrong type")

    def __mul__(self, other):
        if isinstance(other, int):
            empty = InfluenceGrid(self.shape())
            empty.grid = self.grid * other
            return empty
        else:
            raise ValueError("Wrong type")

    def __and__(self, other):
        empty = copy(self)
        empty.grid[other.grid == 0] = 0
        return empty

    def __xor__(self, other):
        empty = InfluenceGrid(self.shape())
        empty.grid = np.copy(self.grid)
        empty.grid[np.nonzero(other.grid)] = 0
        return empty

    def find_best_position(self, size: int) -> Point2:
        pos, = sc2_math.max_sub_matrix(self.grid, size=size)
        return Point2((pos[3] + self.shape()[1], pos[1] + self.shape()[0]))

    def slice_grid(self, dim: float, center_position):
        new_region = copy(self)
        if isinstance(center_position, Unit):
            center_position = center_position.position_tuple

        if (
                round(center_position[0] % 1, 2) == 0.5 and dim % 0 == 0
                or round(center_position[0] % 1, 2) == 0 and dim % 2 == 1
        ):
            dim += 1
        x, y = (round(center_position[1] - dim / 2), round(center_position[0] - dim / 2))
        new_region.grid = new_region.grid[x:(x + dim), y:(y + dim)]
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        new_region.offset = (x, y)
        return new_region
        pass

    def add_points_with_weight(self, weight: int, points):
        for point in points:
            self.grid[point[1], point[0]] += weight

    def set_points_with_weight(self, weight: int, points):
        for point in points:
            self.grid[point[1], point[0]] = weight

    def add_weight(self, weight: int, position: Union[Point2, tuple], radius: float, cycle: bool = False):
        """
        :param weight:
        :param position: Point2 or tuple
        :param radius:
        :param cycle:
        """
        if cycle:
            self.grid[sc2_math.points_in_circle_np(radius, position, self.shape())] += weight
        else:
            self.grid[sc2_math.points_in_square_np(radius, position, self.shape())] += weight

    def set_weight(self, weight: int, position: Union[Point2, tuple], radius: float, cycle: bool = False):
        """
        :param weight:
        :param position: Point2 or tuple
        :param radius:
        :param cycle:
        """
        if cycle:
            self.grid[sc2_math.points_in_circle_np(radius, position, self.shape())] = weight
        else:
            self.grid[sc2_math.points_in_square_np(radius, position, self.shape())] = weight

    def shape(self) -> tuple:
        """
        :return: return shape of the grid
        """
        return self.grid.shape

    def show_grid(self):
        plt.matshow(self.grid)
        plt.show()
        pass

    def on_create(self, bot, map_data=None):
        pass

    def on_update(self):
        pass

    def on_unit_destroyed(self, unit):
        pass

    def on_unit_created(self, unit):
        pass

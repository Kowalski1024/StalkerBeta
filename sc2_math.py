import math
import random
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Set
import itertools as it

from Cython.Includes import numpy
from loguru import logger
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

# Some functions copied from https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/sc2math.py

pi2 = 2 * math.pi


def points_on_circumference(center: Point2, radius, n=10) -> Set[Point2]:
    """Calculates all points on the circumference of a circle. n = number of points."""
    points = [
        (center.x + (math.cos(2 * pi / n * x) * radius), center.y + (math.sin(2 * pi / n * x) * radius))  # x  # y
        for x in range(0, n)
    ]
    point2list = set(map(lambda t: Point2(t), points))
    return point2list


def points_in_square_np(radius, center: Union[tuple, Point2], shape):
    square = np.zeros(shape=shape)
    x_start, y_start = center[1] - radius, center[0] - radius
    x_start = 0 if x_start < 0 else x_start
    y_start = 0 if y_start < 0 else y_start
    square[int(x_start):int(center[1] + radius), int(y_start):int(center[0] + radius)] = 1
    return np.nonzero(square)


def points_in_circle_np(radius, center: Union[tuple, Point2], shape):
    xx, yy = np.ogrid[:shape[0], :shape[1]]
    circle = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
    return np.nonzero(circle <= radius ** 2)


def facing_region(shape, matrix, my_units, to_position, dis: int = 4, num: int = 128, add: bool = False,
                  offset_angle: float = math.pi / 3):
    """
    :param shape:
    :param matrix:
    :param my_units:
    :param to_position:
    :param dis:
    :param num:
    :param add:
    :param offset_angle:
    """

    def region(unit_position):
        angle = math.atan2(to_position[1] - unit_position[1], to_position[0] - unit_position[0])
        if dis < 0:
            angle -= math.pi
        if angle < 0:
            angle += math.pi * 2
        vector_one = (math.cos(angle - offset_angle), math.sin(angle - offset_angle))
        vector_two = (math.cos(angle + offset_angle), math.sin(angle + offset_angle))
        point_list = set()
        for x in np.arange(0.2, abs(dis + 1), 1):
            for y in np.arange(0.2, abs(dis + 1), 1):
                vector_sum = (x * vector_one[0] + y * vector_two[0], x * vector_one[1] + y * vector_two[1])
                y_pos, x_pos = round(unit_position[1] + vector_sum[1]), round(unit_position[0] + vector_sum[0])
                point_list.add(tuple((y_pos, x_pos)))

        for y, x in point_list:
            if 100 < matrix[y][x] != 0:
                if add:
                    matrix[y][x] += num
                else:
                    matrix[y][x] = num

    if matrix.shape != shape:
        logger.debug("Wrong shape")
        return

    if isinstance(to_position, Unit):
        to_position = to_position.position_tuple

    if isinstance(my_units, Point2):
        region(my_units)
    elif isinstance(my_units, Unit):
        region(my_units.position_tuple)
    else:
        for u in my_units:
            region(u.position_tuple)


def find_building_position(matrix: np.ndarray, size: int = 0, min_value: int = 0):
    """
    :param min_value:
    :param matrix:
    :param size:
    """
    # TODO: sum the squares in matrix before searching
    # https://www.techiedelight.com/find-maximum-sum-submatrix-in-given-matrix/

    max_sum = 0
    pos_list = list()
    nonzero = np.nonzero(matrix)
    x_axis = nonzero[0]
    y_axis = nonzero[1]
    for x, y in zip(x_axis, y_axis):
        square_sum = np.sum(matrix[x:(x + size), y:(y + size)])
        if square_sum > min_value:
            if square_sum > max_sum:
                pos_list.clear()
                max_sum = square_sum
                pos_list.append((x-1, y-1))
            elif square_sum == max_sum:
                pos_list.append((x-1, y-1))
    if pos_list:
        pos = random.choice(pos_list)
    else:
        return None
    return Point2((pos[1]+(size+1)/2, pos[0]+(size+1)/2))


# def find_best_position(matrix, size: int = 1):
#     """
#     :param matrix:
#     :param size:
#     :return:
#     """
#     pos = max_square_in_matrix(matrix, size)
#     if pos is None:
#         return None
#     return Point2((pos[3], pos[1]))

import math
import random
from math import pi

import numpy as np
from typing import List, Union, Set
import itertools as it

from Cython.Includes import numpy
from loguru import logger
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

# https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/sc2math.py

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
    mat = np.empty(shape, dtype=bool)

    print(mat)
    return mat
    # return np.nonzero(circle <= radius ** 2)


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


def max_sub_matrix(matrix_in: np.ndarray, size: int = 0):
    """
    :param matrix_in:
    :param size:
    :return [(y_start, y_end, x_start, x_end), max_sum]:
    """
    max_sum = 0
    pos = [0, 0, 0, 0]
    pos_list = list()
    matrix_size = matrix_in.shape
    matrix = np.full(matrix_size, -255)
    matrix += matrix_in
    if size == 0:
        # Kadane algorithm
        for left in range(matrix_size[0]):
            for right in range(left, matrix_size[0]):
                sub_matrix = matrix[:matrix_size[1] + 1, left:right + 1]
                side_array = np.sum(sub_matrix, axis=1)
                sum_arr = 0
                start = 0
                end = 0
                while end < side_array.shape[0]:
                    sum_arr += side_array[end]
                    if sum_arr < 0:
                        sum_arr = 0
                        start = end + 1
                    elif sum_arr > max_sum:
                        max_sum = sum_arr
                        pos = [start, end, left, right]
                    elif (
                            sum_arr == max_sum
                            and size > 1
                            and (pos[1] - pos[0]) * (pos[3] - pos[2]) < (right - left) * (end - start)
                    ):
                        max_sum = sum_arr
                        pos = [start, end, left, right]
                    end += 1
    else:
        size -= 1
        for x in range(matrix_size[0] - size):
            for y in range(matrix_size[1] - size):
                sub_matrix = matrix[x:(x + size + 1), y:(y + size + 1)]
                sum_arr = np.sum(sub_matrix)
                if sum_arr >= 0:
                    if sum_arr > max_sum:
                        pos_list.clear()
                        max_sum = sum_arr
                        pos_list.append([x, x + size, y, y + size])
                    elif sum_arr == max_sum:
                        pos_list.append([x, x + size, y, y + size])
    if pos_list:
        pos = random.choice(pos_list)
    return [pos, max_sum]

import math
import random
from math import pi
import numpy as np
from typing import List, Union, Set, Tuple
from collections import deque
from itertools import chain
import scipy.spatial

from loguru import logger
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

# Some functions copied from https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/sc2math.py

pi2 = 2 * math.pi


class Segment:
    def __init__(self, p1: tuple, p2: tuple):
        self.p1 = p1
        self.p2 = p2
        self.vector = (p2[0] - p1[0], p2[1] - p1[1])

    def __repr__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"

    def intersection(self, other) -> list:
        if isinstance(other, Segment):
            det = (-other.vector[0] * self.vector[1] + self.vector[0] * other.vector[1])
            if det == 0:
                return []
            s, t = self._get_parameters(other, det)
            if 0 <= s <= 1 and 0 <= t <= 1:
                return [(self.p1[0] + (t * self.vector[0]), self.p1[1] + (t * self.vector[1]))]
        elif isinstance(other, LinearRing):
            return other.intersection(self)
        return []

    def intersects(self, other) -> bool:
        if isinstance(other, Segment):
            det = (-other.vector[0] * self.vector[1] + self.vector[0] * other.vector[1])
            if det == 0:
                return False
            s, t = self._get_parameters(other, det)
            if 0 <= s <= 1 and 0 <= t <= 1:
                return True
        elif isinstance(other, LinearRing):
            return other.intersects(self)
        else:
            raise KeyError('Only implemented intersection of other Segment or LinearRing')
        return False

    def _get_parameters(self, other, det):
        s = (-self.vector[1] * (self.p1[0] - other.p1[0]) + self.vector[0] * (self.p1[1] - other.p1[1])) / det
        t = (other.vector[0] * (self.p1[1] - other.p1[1]) - other.vector[1] * (self.p1[0] - other.p1[0])) / det
        return s, t


class LinearRing:
    def __init__(self, vertices: List[tuple]):
        self.vertices = vertices
        ver = vertices.copy()
        ver.append(ver.pop(0))
        self.segment_list: List[Segment] = [Segment(p1, p2) for p1, p2 in zip(vertices, ver)]

    def intersects(self, other: Segment) -> bool:
        if isinstance(other, Segment):
            for segment in self.segment_list:
                if segment.intersects(other):
                    return True
        else:
            raise KeyError('Only implemented Segment intersection')
        return False

    def intersection(self, other: Segment) -> list:
        intersection_list = list()
        if isinstance(other, Segment):
            for segment in self.segment_list:
                intersection_list.extend(segment.intersection(other))
        else:
            raise KeyError('Only implemented Segment intersection')
        return intersection_list


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
                pos_list.append((x - 1, y - 1))
            elif square_sum == max_sum:
                pos_list.append((x - 1, y - 1))
    if pos_list:
        pos = random.choice(pos_list)
    else:
        return None
    return Point2((pos[1] + (size + 1) / 2, pos[0] + (size + 1) / 2))


def vectors_angle(vector_1: Point2, vector_2: Point2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def distance_point2_to_line(p: Point2, line: Tuple[Point2, Point2]):
    return np.linalg.norm(
        np.cross(line[0] - line[1], line[1] - p)) / np.linalg.norm(
        line[1] - line[1])


def dfs_numpy(center: tuple, array: np.ndarray) -> np.ndarray:
    center = int(center[0]), int(center[1])
    deq = deque([center])
    polygon_points = np.zeros(array.shape, dtype=bool)
    while deq:
        x, y = deq.popleft()
        if array[y][x]:
            if array[y][x] == 1:
                deq.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            array[y][x] = False
            polygon_points[y][x] = True
    return polygon_points


def points2_to_indices(points: Set[Point2]) -> Tuple[np.ndarray, np.ndarray]:
    return np.array([p[1] for p in points]), np.array([p[0] for p in points])


def indices_to_points2(indices) -> Set[Point2]:
    return {Point2((x, y)) for y, x in zip(*indices)}


def ndarray_corners_points2(array: np.ndarray):
    def neighbors4(x, y) -> set:
        return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}

    nonzero = {(x, y) for x, y in zip(*np.nonzero(array))}
    return {Point2((y, x)) for x, y in nonzero if neighbors4(x, y).difference(nonzero)}


def point2_in_grid(point: Point2, grid: np.ndarray) -> bool:
    return bool(grid[int(point.y)][int(point.x)])


# copied from MapAnalyzer
def closest_towards_point(points: List[Point2], target: Union[Point2, tuple]) -> Point2:

    if not isinstance(points, (list, np.ndarray)):
        logger.warning(type(points))

    return points[closest_node_idx(node=target, nodes=points)]


# copied from MapAnalyzer
def closest_node_idx(node: Union[Point2, np.ndarray], nodes: Union[List[Tuple[int, int]], np.ndarray]) -> int:
    """
    :rtype: int

    Given a list of ``nodes``  and a single ``node`` ,

    will return the index of the closest node in the list to ``node``

    """
    if isinstance(nodes, list):
        iter = chain.from_iterable(nodes)
        nodes = np.fromiter(iter, dtype=type(nodes[0][0]), count=len(nodes) * 2).reshape((-1, 2))

    closest_index = scipy.spatial.distance.cdist([node], nodes, "sqeuclidean").argmin()
    return closest_index

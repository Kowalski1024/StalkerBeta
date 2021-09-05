from .connectivity import ConnectivityLine
from sc2.position import Point2, Point3
from typing import Union, List, Tuple, Set, Dict
import numpy as np


class Polygon:
    def __init__(self, arr: np.ndarray):
        self.has_base = False
        self.array = arr
        self.points = {Point2((y, x)) for x, y in zip(*np.nonzero(arr))}
        self.connectivity_dict: Dict[Polygon, List[ConnectivityLine]] = dict()

    def add_point(self, p: Union[Point2, tuple]):
        self.array[p[1]][p[0]] = True
        self.points.add(p)

    def add_points(self, points: Union[set, list]):
        for p in points:
            self.add_point(p)

    def add_polygon(self, poly: 'Polygon'):
        self.array[poly.array] = True
        self.points.update(poly.points)

    def connect_polygon(self, connect_line: ConnectivityLine, no_exists=False):
        if connect_line.parent != self:
            connect_line.neighbour, connect_line.parent = connect_line.parent, connect_line.neighbour
        if no_exists and connect_line.neighbour in self.connectivity_dict:
            return
        if connect_line.neighbour not in self.connectivity_dict:
            self.connectivity_dict[connect_line.neighbour] = [connect_line]
        else:
            self.connectivity_dict[connect_line.neighbour].append(connect_line)

    @property
    def area(self) -> int:
        return len(self.points)


class BaseRegion(Polygon):
    def __init__(self, arr: np.ndarray, base: Point2):
        super().__init__(arr)
        self.base_location = base
        self.has_base = True

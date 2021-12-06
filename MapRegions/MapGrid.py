import numpy as np
from typing import List, Set, Tuple, Optional, Union
import matplotlib.pyplot as plt

from sc2.position import Point2
from sc2.units import Units

from .utils import change_destructable_status_in_grid


class MapGrid:
    def __init__(self, grid: np.ndarray):
        self._grid: np.ndarray = grid.copy()

    def is_inside_point(self, point: Point2) -> bool:
        point = point.rounded
        return self._grid[point.y, point.x] != 0

    @property
    def grid(self):
        return self._grid

    def include_grid(self, grid: np.ndarray, status: int):
        self._grid[np.nonzero(grid)] = status

    def include_points(self, points, status: int):
        for p in points:
            self._grid[p.y, p.x] = status

    def include_destructables(self, units: Units, status: int):
        grid = self._grid.T
        for unit in units:
            change_destructable_status_in_grid(grid, unit, status)
        self._grid = grid.T

    def bfs_points(self,
                   point: Point2,
                   max_value: int = 0,
                   sub: bool = False,
                   neighbour8: bool = False
                   ) -> Tuple[Set[Point2], Set[Point2]]:

        points = set()
        surroundings = set()
        deq = [point.rounded]

        while deq:
            p = deq.pop(0)
            val = self._grid[p.y, p.x]
            if val:
                points.add(p)
                if max_value == 0 or val <= max_value:
                    deq.extend(p.neighbors8) if neighbour8 else deq.extend(p.neighbors4)
                if sub:
                    self._grid[p.y, p.x] = False
            else:
                surroundings.add(p)
        return points, surroundings.difference(points)

    def bfs_ndarray(self, point: Point2, max_value: int = 0, sub: bool = False) -> np.ndarray:
        grid = np.zeros(self._grid.shape, dtype=bool)
        deq = [point.rounded]

        while deq:
            p = deq.pop(0)
            val = self._grid[p.y, p.x]
            if val:
                grid[p.y, p.x] = True
                if max_value == 0 or val <= max_value:
                    deq.extend(p.neighbors4)
                if sub:
                    self._grid[p.y, p.x] = False
        return grid

    def plot_grid(self):
        plt.matshow(self._grid)
        plt.show()

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
from sc2.position import Point2
from sc2.game_info import GameInfo

import matplotlib.pyplot as plt


@dataclass
class Point:
    x: int
    y: int
    size: int
    axis: str


class WallBuilder:
    def __init__(self, choke_points: Set[Point2], game_info: GameInfo):
        self._game_info = game_info
        self.choke_points = choke_points
        self.shape, self.min_p, self.max_p = self.borders()
        self.placement_grid = np.transpose(game_info.placement_grid.data_numpy.copy())
        self.placement_grid = self.placement_grid[self.min_p.x - 2:self.max_p.x + 3, self.min_p.y - 2:self.max_p.y + 3]
        self.pathing_grid = np.transpose(game_info.pathing_grid.data_numpy.copy())

    def borders(self) -> Tuple[Point2, Point2, Point2]:
        max_x, max_y = max(self.choke_points, key=lambda x: x[0])[0], max(self.choke_points, key=lambda x: x[1])[1]
        min_x, min_y = min(self.choke_points, key=lambda x: x[0])[0], min(self.choke_points, key=lambda x: x[1])[1]
        return (Point2((max_x - min_x + 1, max_y - min_y + 1)),
                Point2((min_x, min_y)),
                Point2((max_x, max_y)))

    def wall_search(self, units):
        is_rotated = False
        if not (self.min_p in self.choke_points and any(self.pathing_grid[x] == 0 for x in self.min_p.neighbors4)):
            self.placement_grid = np.rot90(self.placement_grid)
            self.shape = Point2((self.shape.y, self.shape.x))
            is_rotated = True
        ndarray = np.zeros(self.placement_grid.shape)
        for p in self.wall_first(units):
            ndarray[p[0]+2, p[1]+2] = p[2]
        if is_rotated:
            ndarray = np.rot90(ndarray, k=3)
        return [(Point2((a, b)) + self.min_p, val) for (a, b), val in np.ndenumerate(ndarray) if val > 0]

    def neighbours(self, p: Point) -> List[Point]:
        res = []
        if p.size == 1:
            res = [Point(p.x + 1, p.y, 1, p.axis)] if p.axis == 'x' else [
                Point(p.x, p.y + 1, 1, p.axis)]
        elif p.size == 2:
            res = [Point(p.x - 1, p.y + 1, 2, 'y'), Point(p.x, p.y + 1, 2, 'y'),
                   Point(p.x + 1, p.y - 1, 2, 'x'), Point(p.x + 1, p.y, 2, 'x')]
        elif p.size == 3:
            res = [Point(p.x - 1, p.y + 2, 3, 'y'), Point(p.x, p.y + 2, 3, 'y'), Point(p.x + 1, p.y + 2, 3, 'y'),
                   Point(p.x + 2, p.y - 1, 3, 'x'), Point(p.x + 2, p.y, 3, 'x'), Point(p.x + 2, p.y + 1, 3, 'x')]

        res = [item for item in res if
               0 <= item.x < self.shape[0]
               and 0 <= item.y < self.shape[1]
               and self.placement_grid[item.x + 2, item.y + 2]]
        return res

    @staticmethod
    def placement(pos: Tuple[int, int], size) -> List[Tuple[int, int]]:
        if size == 1:
            return [pos]
        elif size == 2:
            return [(pos[0] + 1, pos[1] + 1), (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1), pos]
        elif size == 3:
            return [(pos[0] + x, pos[1] + y) for x in [1, 0, -1] for y in [1, 0, -1]]

    def wall_first(self, units_radius: List[int]):
        shape = self.shape
        res = []
        for idx, rad in enumerate(units_radius):
            for x, y in self.placement((0, 0), rad):
                if self.can_place((x, y), rad):
                    if self.can_place((x, y), rad):
                        res = self.wall(units_radius, {idx}, Point(x, y, rad, 'x'), shape)
                    if res:
                        break
            if res:
                break
        if res is None:
            res = []
        return res

    def wall(self, units_radius: list, used_units: set, prev: Point, shape: Point2) -> Optional[list]:
        res = None
        if shape.x < 0 and shape.y < 0:
            return None
        elif shape.x <= 0 and shape.y == 0 or shape.x == 0 and shape.y <= 0:
            return [(prev.x, prev.y, prev.size)]
        unused_units = [i for i in range(len(units_radius)) if i not in used_units]
        ps = self.neighbours(prev)
        for idx in unused_units:
            rad = units_radius[idx]
            for p in ps:
                if rad == 1:
                    sub = Point2((1, 0)) if p.axis == 'x' else Point2((0, 1))
                    if self.can_place((p.x, p.y), rad):
                        res = self.wall(units_radius, used_units | {idx}, Point(p.x, p.y, 1, p.axis), shape - sub)
                else:
                    for k in range(1, 1 - rad, -1):
                        if p.axis == 'x':
                            if self.can_place((p.x + 1, p.y + k), rad):
                                sub = Point2((p.x + 2 + rad % 2, p.y + k + rad % 2 + 1))
                                res = self.wall(units_radius, used_units | {idx}, Point(p.x + 1, p.y + k, rad, 'x'),
                                                self.shape - sub)
                        else:
                            if self.can_place((p.x + k, p.y + 1), rad):
                                sub = Point2((p.x + k + rad % 2 + 1, p.y + 2 + rad % 2))
                                res = self.wall(units_radius, used_units | {idx}, Point(p.x + k, p.y + 1, rad, 'y'),
                                                self.shape - sub)
                        if res is not None:
                            break
                if res is not None:
                    res.append((prev.x, prev.y, prev.size))
                    return res
        return res

    def can_place(self, pos, k):
        if k == 1:
            return self.placement_grid[pos[0] + 2, pos[1] + 2]
        kernel = np.ones((k, k))
        sum_ = convolve2d(self.placement_grid, kernel, mode='same')
        return sum_[pos[0] + 2, pos[1] + 2] >= k ** 2

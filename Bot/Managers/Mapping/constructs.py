from sc2.position import Point2, Point3
from typing import Union, List, Tuple, Set, Dict
from sc2.game_info import GameInfo
from sc2.cache import property_immutable_cache, property_mutable_cache


class Ramp:
    def __init__(self, points: Set[Point2], game_info: GameInfo):
        self._points = points
        self.__game_info = game_info

        self.cache = {}

    @property_immutable_cache
    def _height_map(self):
        return self.__game_info.terrain_height

    def height_at(self, p: Point2) -> int:
        return self._height_map[p]

    def heights(self):
        return sorted(list({self.height_at(point) for point in self._points}))

    @property_mutable_cache
    def points(self) -> Set[Point2]:
        return self._points.copy()

    @property_mutable_cache
    def centers(self) -> List[Point2]:
        height_points = self.heights()
        mid = len(height_points) / 2
        if len(height_points) > 2:
            top_height = height_points[int(mid + 1)]
            bottom_height = height_points[int(mid - 1)]
        else:
            top_height = height_points[0]
            bottom_height = height_points[1]
        upper = []
        down = []
        for p in self._points:
            h = self.height_at(p)
            if h == top_height:
                upper.append(p)
            elif h == bottom_height:
                down.append(p)
        u, d = Point2.center(upper), Point2.center(down)
        return [u, Point2.center([u, d]), d]

    def upper_side_point(self, distance=8):
        top_center, center, down_center = self.centers
        vec = (down_center - center).normalized
        return -distance * vec + center

    def down_side_point(self, distance=8):
        top_center, center, down_center = self.centers
        vec = (down_center - center).normalized
        return distance * vec + center

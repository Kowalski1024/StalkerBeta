from sc2.position import Point2, Point3
from typing import TYPE_CHECKING, Union, List, Tuple, Set, Dict
from sc2.game_info import GameInfo

from sc2.units import Units
from sc2.cache import property_immutable_cache, property_mutable_cache
from collections import deque
if TYPE_CHECKING:
    from sc2.bot_ai import BotAI

# copied from sc2.cache.py (BurnySC2)
def _property_cache_once_per_frame(f):
    """This decorator caches the return value for one game loop,
    then clears it if it is accessed in a different game loop."""

    from functools import wraps

    @wraps(f)
    def inner(self):
        property_cache = "_cache_" + f.__name__
        state_cache = "_frame_" + f.__name__
        cache_updated = getattr(self, state_cache, -1) == self._bot.state.game_loop
        if not cache_updated:
            setattr(self, property_cache, f(self))
            setattr(self, state_cache, self._bot.state.game_loop)

        cache = getattr(self, property_cache)
        should_copy = callable(getattr(cache, "copy", None))
        if should_copy:
            return cache.copy()
        return cache

    return property(inner)


class Ramp:
    def __init__(self, points: Set[Point2], game_info: 'GameInfo'):
        self._points = points
        self.__game_info = game_info

        self.cache = {}

    @property_immutable_cache
    def _height_map(self):
        return self.__game_info.terrain_height

    def height_at(self, p: Point2) -> int:
        return self._height_map[p.rounded]

    def heights(self):
        return sorted(list({self.height_at(point) for point in self._points}))

    def _passable(self, p: Point2):
        return self.__game_info.pathing_grid[p] == 1

    def _buildable(self, p: Point2):
        return self.__game_info.placement_grid[p] == 1

    @property_mutable_cache
    def points(self) -> Set[Point2]:
        return self._points.copy()

    @property
    def upper_center(self) -> Point2:
        return Point2.center(self.upper)

    @property
    def lower_center(self) -> Point2:
        return Point2.center(self.lower)

    @property_mutable_cache
    def upper(self):
        ps = set()
        for p in self._points:
            ps.update(p.neighbors8)
        ps = {p for p in ps if self._buildable(p)}
        h = max([self.height_at(x) for x in ps])
        # some points have wrong height
        for p in ps:
            if self.height_at(p) == h and not any(self.height_at(x) != h for x in p.neighbors8.intersection(ps)):
                return self.find_group(p, ps)
        return {p for p in ps if self.height_at(p) == h}

    @property_mutable_cache
    def lower(self) -> Set[Point2]:
        ps = set()
        for p in self._points:
            ps.update(p.neighbors8)
        ps = {p for p in ps if self._buildable(p)}
        h = min([self.height_at(x) for x in ps])
        # some points have wrong height
        for p in ps:
            if self.height_at(p) == h and not any(self.height_at(x) != h for x in p.neighbors8.intersection(ps)):
                return self.find_group(p, ps)
        return {p for p in ps if self.height_at(p) == h}

    def upper_side_point(self, distance=6):
        vec = self.lower_center.direction_vector(self.upper_center).normalized
        return distance * vec + self.upper_center

    def lower_side_point(self, distance=6):
        vec = self.upper_center.direction_vector(self.lower_center).normalized
        return distance * vec + self.lower_center

    @staticmethod
    def find_group(p: Point2, ps: Set[Point2]) -> Set[Point2]:
        s = set()
        visited = set()
        deq = deque([p])
        while deq:
            p = deq.popleft()
            if p in ps and p not in visited:
                visited.add(p)
                s.add(p)
                deq.extend(p.neighbors8)
        return s


class Blocker:
    def __init__(self, bot: 'BotAI', units: Units):
        self.tags = units.tags
        self._bot = bot
        self.is_minerals = bool(self._bot.mineral_field.tags_in(self.tags))
        self.is_rock = bool(self._bot.destructables.tags_in(self.tags))
        self.cache = {}

    @_property_cache_once_per_frame
    def get_blockers(self) -> Units:
        blockers = Units([], self._bot)
        if not self.tags:
            return blockers
        blockers.extend(self._bot.destructables.tags_in(self.tags))
        if blockers:
            self.tags = {u.tag for u in blockers}
            return blockers
        blockers.extend(self._bot.mineral_field.tags_in(self.tags))
        self.tags = {u.tag for u in blockers}
        return blockers

    @property
    def exists(self):
        if not self.tags:
            return False
        else:
            return len(self.get_blockers) > 0

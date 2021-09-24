from typing import TYPE_CHECKING, Set, List, Union, Optional, Tuple
from sc2.position import Point2
from sc2.units import Units
from sc2.cache import property_immutable_cache, property_mutable_cache

if TYPE_CHECKING:
    from MapRegions import MapRegions, Region, Blocker


class ConnectivitySide:
    def __init__(self, map_regions: 'MapRegions',
                 reg: 'Region',
                 neighbour_side: Optional['ConnectivitySide'],
                 points: Set[Point2],
                 impassable, jumpable,
                 blocker: Optional['Blocker'] = None
                 ):
        self.cache = {}

        self._map_regions = map_regions
        self.region = reg
        self.neighbour_side = neighbour_side
        self.points = points

        self.blocker = blocker
        self.impassable = impassable
        self.jumpable = jumpable

    @property
    def blocker_exists(self):
        return False if self.blocker is None else self.blocker.exists

    @property
    def neighbour(self):
        return self.neighbour_side.region

    @property
    def passable(self):
        return not (self.impassable or self.blocker_exists)

    @property_immutable_cache
    def center(self) -> Point2:
        c1 = Point2.center(self.points)
        c2 = Point2.center(self.neighbour_side.points)
        vec = (c1-c2).normalized
        return 4*vec + c1

    @property_immutable_cache
    def edge_line(self):
        cen = Point2.center(self.points)
        ps = self.points.copy()
        max1 = max(ps, key=lambda x: cen.distance_to_point2(x))
        ps.remove(max1)
        max2 = max(ps, key=lambda x: cen.distance_to_point2(x))
        return max1, max2


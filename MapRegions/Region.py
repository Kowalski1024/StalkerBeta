import sc2math
from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.unit import Unit
from typing import TYPE_CHECKING, Union, List, Tuple, Set, Dict, Optional
import numpy as np
from functools import lru_cache
from scipy.ndimage import center_of_mass
from sc2.cache import property_immutable_cache, property_mutable_cache

if TYPE_CHECKING:
    from MapRegions import ConnectivitySide


class Region:
    def __init__(self, arr: np.ndarray, expansions: List[Point2], watchtowers: Units):
        self.cache = {}

        self.array = arr
        self.indices = np.nonzero(self.array)
        self.points = sc2math.indices_to_points2(self.indices)
        self.bases = [
            base
            for base in expansions
            if self.is_inside_point(base)
        ]
        self.watchtowers = [
            tower
            for tower in watchtowers
            if self.is_inside_point(tower.position_tuple)
        ]
        self.label = None
        self.connectivity_dict: Dict[int, List[ConnectivitySide]] = dict()

    def _update(self, reg: 'Region'):
        # used in regions calculation when small polygon are merged to neighbours
        self.array[reg.array] = True
        self.indices = np.nonzero(self.array)
        self.points.update(reg.points)
        self.bases.extend(reg.bases)
        self.watchtowers.extend(reg.watchtowers)

    # def connect_polygon(self, connect_line: ConnectivityLine, no_exists=False):
    #     if connect_line.parent != self:
    #         connect_line.neighbour, connect_line.parent = connect_line.parent, connect_line.neighbour
    #     if no_exists and connect_line.neighbour in self.connectivity_dict:
    #         return
    #     if connect_line.neighbour not in self.connectivity_dict:
    #         self.connectivity_dict[connect_line.neighbour] = [connect_line]
    #     else:
    #         self.connectivity_dict[connect_line.neighbour].append(connect_line)

    @property
    def has_base(self) -> bool:
        return len(self.bases) > 0

    @property
    def has_watchtower(self) -> bool:
        return len(self.watchtowers) > 0

    @property
    def base_position(self) -> Point2:
        return self.bases[0]

    @property
    def watchtower(self) -> Unit:
        return self.watchtowers[0]

    @property_immutable_cache
    def center(self) -> Point2:
        """

        Since the center is always going to be a ``float``,

        and for performance considerations we use integer coordinates.

        We will return the closest point registered

        """

        cm = sc2math.closest_towards_point(points=list(self.points), target=center_of_mass(self.array))
        return cm

    @lru_cache()
    def is_inside_point(self, point: Union[Point2, tuple]) -> bool:
        """

        Query via Set(Point2)  ''fast''

        """
        if isinstance(point, Point2):
            point = point.rounded
        if point in self.points:
            return True
        return False

    @lru_cache()
    def is_inside_indices(self, point: Union[Point2, tuple]) -> bool:
        """

        Query via 2d np.array  ''slower''

        """
        if isinstance(point, Point2):
            point = point.rounded
        return point[1] in self.indices[0] and point[0] in self.indices[1]

    @property
    @lru_cache()
    def perimeter_as_points2(self) -> Set[Point2]:
        return sc2math.ndarray_corners_points2(self.array)

    @property
    def perimeter_as_indices(self):
        return sc2math.points2_to_indices(self.perimeter_as_points2)

    @property
    def top(self):
        return max(self.points, key=lambda x: (x[1], 0))

    @property
    def bottom(self):
        return min(self.points, key=lambda x: (x[1], 0))

    @property
    def right(self):
        return max(self.points, key=lambda x: (x[0], 0))

    @property
    def left(self):
        return min(self.points, key=lambda x: (x[0], 0))

    @property
    def area(self) -> int:
        return len(self.points)

    def plot_perimeter(self, self_only: bool = True):
        """

        Debug Method plot_perimeter

        """
        import matplotlib.pyplot as plt

        x, y = self.perimeter_as_indices
        plt.scatter(x, y, s=0.1)
        if self_only:  # pragma: no cover
            plt.grid()

    def __repr__(self) -> str:
        return f"Region {self.label} [size={self.area}, bases={len(self.bases)}]"

from typing import TYPE_CHECKING, Set
from sc2.position import Point2

if TYPE_CHECKING:
    from .polygon import Polygon


class ConnectivityLine:
    def __init__(self, parent: 'Polygon', neighbour: 'Polygon'):
        self.parent = parent
        self.neighbour = neighbour
        self.points: Set[Point2] = set()

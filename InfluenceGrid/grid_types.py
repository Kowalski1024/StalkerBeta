from enum import Enum


class AllyGrid(Enum):
    StructureGrid = 0
    PowerGrid = 1


class EnemyGrid(Enum):
    GroundDmg = 0
    AirDmg = 1
    Vision = 2

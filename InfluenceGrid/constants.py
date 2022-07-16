from enum import Enum


class PylonGrid(float, Enum):
    PoweredStructure = 0.3
    UnpoweredStructure = 2
    PlannedStructures = 2.5
    Townhall = 0.5


class StructureGrid(float, Enum):
    PowerGrid = 0.2


class BuildableGrid(float, Enum):
    MainRegion = 16
    OtherRegions = 14
    Adjacency = 0.1


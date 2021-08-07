from .manager_base import ManagerBase
from MapInfluence.grid_types import GridTypes
from MapInfluence.influence_grid import InfluenceGrid
from MapInfluence.basic_grids import PlacementGrid, PowerGrid, BuildingGrid, NaturalGrid
from MapInfluence.advanced_grids import PylonsGrid, StructuresGrid

from sc2.unit import Unit
from sc2.constants import *

from typing import Dict, TYPE_CHECKING
from MapAnalyzer.MapData import MapData

import timeit


class InfluenceGridManager(ManagerBase):
    def __init__(self, bot):
        super().__init__()
        self._bot = bot
        self._map_shape = tuple
        self.grid_dict: Dict[GridTypes, InfluenceGrid] = dict()

    def __getitem__(self, i: GridTypes):
        return self.grid_dict[i]

    def on_create(self, map_data: MapData):
        shape = self._bot.game_info.map_size
        self._map_shape = (shape[1], shape[0])

        # basic grids
        self.grid_dict[GridTypes.Placement] = PlacementGrid(self._map_shape)
        self.grid_dict[GridTypes.Power] = PowerGrid(self._map_shape)
        self.grid_dict[GridTypes.Buildings] = BuildingGrid(self._map_shape)
        self.grid_dict[GridTypes.Natural] = NaturalGrid(self._map_shape)

        # advanced grids
        self.grid_dict[GridTypes.Pylons] = PylonsGrid(self._map_shape)
        self.grid_dict[GridTypes.Structures] = StructuresGrid(self._map_shape)

        self.grid_dict[GridTypes.Placement].on_create(self._bot)
        self.grid_dict[GridTypes.Buildings].on_create(self._bot)
        self.grid_dict[GridTypes.Natural].on_create(self._bot)

        self.grid_dict[GridTypes.Pylons].on_create(self._bot, map_data)
        self.grid_dict[GridTypes.Structures].on_create(self._bot)

    async def update(self):
        pass

    async def post_update(self):
        pass

    async def on_unit_created(self, unit: Unit):
        pass

    async def on_unit_destroyed(self, unit: Unit):
        pass

    async def on_building_construction_started(self, unit: Unit):
        self.grid_dict[GridTypes.Buildings].on_unit_created(unit)
        pass

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id == UnitTypeId.PYLON:
            self.grid_dict[GridTypes.Power].on_unit_created(unit)
        pass

    def get_pylon_grid(self) -> InfluenceGrid:
        output_grid = (self.grid_dict[GridTypes.Pylons] + self.grid_dict[GridTypes.Placement] - self.grid_dict[
            GridTypes.Power]) & self.grid_dict[GridTypes.Buildings] & self.grid_dict[
                   GridTypes.Natural]
        # town_halls without pylon
        pylons = self._bot.structures(UnitTypeId.PYLON)
        town_halls = self._bot.structures(UnitTypeId.NEXUS)
        for town_hall in town_halls:
            if not pylons.closer_than(6.5, town_hall):
                output_grid.add_weight(1, town_hall.position_tuple, 6.5, True)
        end = timeit.default_timer()
        return output_grid

    def get_building_grid(self) -> InfluenceGrid:
        return ((self.grid_dict[GridTypes.Structures] + self.grid_dict[GridTypes.Power]) & self.grid_dict[
            GridTypes.Power] & self.grid_dict[GridTypes.Buildings] & self.grid_dict[
                   GridTypes.Natural] & self.grid_dict[GridTypes.Placement]) + self.grid_dict[GridTypes.Placement]

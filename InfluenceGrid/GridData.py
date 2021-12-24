from typing import DefaultDict, Dict
from collections import defaultdict
import numpy as np
from sc2.bot_ai import BotAI
from sc2.constants import UnitTypeId
from MapRegions.MapRegions import MapRegions

from InfluenceGrid.basic_grid import BasicGrid
from MapRegions.utils import change_destructable_status_in_grid


class GridData:
    def __init__(self, bot: BotAI, map_regions: MapRegions):
        self.map_shape = bot.game_info.map_size
        self._bot = bot
        self._map_regions = map_regions
        self._placement_grid = self._bot.game_info.placement_grid.data_numpy.copy().T
        self.buildable_regs = []

    def pylon_grid(self, regs, main):

        pass

    def power_grid(self):
        grid = BasicGrid(self.map_shape, base_val=0, dtype=np.float32)
        for pylon in self._bot.structures:
            if pylon.type_id == UnitTypeId.PYLON:
                grid.add_circle(1, radius=6.5, center=pylon.position)
        return grid

    def buildable_grid(self, reg_labels, main) -> BasicGrid:
        grid = BasicGrid(self.map_shape, base_val=-256, dtype=np.float32, ndarray=self._placement_grid)
        grid.set_polygon(16, self._map_regions.regions[main].ndarray)
        for label in reg_labels:
            reg = self._map_regions.regions[label]
            if label != main:
                grid.set_polygon(14, reg.ndarray)
        for structure in self._bot.structures | self._bot.enemy_structures:
            grid.set_square(-256, radius=structure.radius, center=structure.position)
        grid.set_points(64, [self._bot.main_base_ramp.protoss_wall_pylon])
        return grid

    def _neutral_grid(self):
        grid = BasicGrid(self.map_shape, base_val=0, dtype=bool)
        for destructable in self._bot.destructables:
            change_destructable_status_in_grid(grid=grid.ndarray, unit=destructable, status=-256)
        for mineral in self._bot.mineral_field:
            pos = mineral.position
            x = int(pos[0]) - 1
            y = int(pos[1])
            grid.ndarray[x:(x + 2), y] = -256
        return grid

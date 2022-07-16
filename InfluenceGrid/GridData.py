from typing import DefaultDict, Dict
from collections import defaultdict
import numpy as np
import skimage.draw

from sc2.bot_ai import BotAI
from sc2.constants import UnitTypeId
from MapRegions.MapRegions import MapRegions

from InfluenceGrid.cache import cache_once_per_frame
from InfluenceGrid.basic_grid import BasicGrid
from InfluenceGrid.constants import PylonGrid, StructureGrid, BuildableGrid
from MapRegions.utils import change_destructable_status_in_grid


class GridData:
    def __init__(self, bot: BotAI, map_regions: MapRegions):
        self.cache = {}

        self.map_shape = bot.game_info.map_size
        self._bot = bot
        self._map_regions = map_regions
        self._placement_grid = self._bot.game_info.placement_grid.data_numpy.copy().T

    def pylon_grid(self, regs_labels, main):
        grid = self.buildable_grid(regs_labels, main)

        # Construct Pylon near nexus, large number of structures, unpowered structures
        for structure in self._bot.structures:
            if structure.type_id == UnitTypeId.NEXUS:
                grid.add_circle(PylonGrid.Townhall.value, 7, structure.position)
            elif structure.type_id not in {UnitTypeId.ASSIMILATOR, UnitTypeId.PYLON}:
                if structure.is_powered:
                    grid.add_circle(PylonGrid.PoweredStructure.value, 6.5, structure.position)
                else:
                    grid.add_circle(PylonGrid.UnpoweredStructure.value, 6.5, structure.position)

        # Avoid the cluster of multiple Pylons
        power_grid = self.power_grid()
        grid -= power_grid

        # Prioritize region near planned structures
        for p in self.structure_grid(regs_labels, main).convolve(3).argsmax(4):
            if not power_grid.ndarray[p]:
                grid.add_circle(PylonGrid.PlannedStructures.value, 6.5, p)
            grid.set_square(-256, 1.5, p)

        return grid

    def structure_grid(self, regs_labels, main):
        grid = self.buildable_grid(regs_labels, main)

        # Prefer region with power
        grid += self.power_grid()*StructureGrid.PowerGrid.value
        return grid

    def buildable_grid(self, reg_labels, main) -> BasicGrid:
        grid = BasicGrid(self.map_shape, base_val=-256, dtype=np.float32, ndarray=self._placement_grid)

        # Prefer regions provided in params
        for label in reg_labels:
            grid.set_polygon(BuildableGrid.OtherRegions.value, self._map_regions.regions[label].ndarray)
        grid.set_polygon(BuildableGrid.MainRegion.value, self._map_regions.regions[main].ndarray)

        # Prefer to construct a building adjacent to other structure
        for structure in self._bot.structures:
            grid.set_square(-256, radius=structure.radius, center=structure.position)
            grid.add_square(BuildableGrid.Adjacency.value, structure.radius + 1, structure.position)
        for structure in self._bot.enemy_structures:
            grid.set_square(-256, radius=structure.radius, center=structure.position)
        grid.set_polygon(-256, self._neutral_grid().ndarray)
        return grid

    def power_grid(self):
        grid = BasicGrid(self.map_shape)
        for pylon in self._bot.structures:
            if pylon.type_id == UnitTypeId.PYLON:
                grid.add_circle(1, radius=6.5, center=pylon.position)
        return grid

    def _neutral_grid(self):
        grid = BasicGrid(self.map_shape, base_val=0, dtype=bool)
        for destructable in self._bot.destructables:
            change_destructable_status_in_grid(grid=grid.ndarray, unit=destructable, status=1)
        for mineral in self._bot.mineral_field:
            pos = mineral.position
            x = int(pos[0]) - 1
            y = int(pos[1])
            grid.ndarray[x:(x + 2), y] = 1
        for geyser in self._bot.vespene_geyser:
            grid.set_square(1, 1.5, geyser.position)
        for loc in self._map_regions.expansion_locations:
            resources = self._bot.resources.filter(lambda unit: unit.distance_to(loc) <= 10)
            for res in resources:
                rr, cc = skimage.draw.line(*loc.rounded, *res.position.rounded)
                grid.ndarray[rr, cc] = 1
        return grid

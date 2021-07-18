from MapInfluence.influence_grid import InfluenceGrid

from sc2.bot_ai import BotAI
from sc2.unit import Unit
from sc2.constants import *


class PlacementGrid(InfluenceGrid):
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def on_create(self, bot, map_data=None):
        self.grid = bot.game_info.placement_grid.data_numpy
        pass


class PowerGrid(InfluenceGrid):
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def on_create(self, bot, map_data=None):
        pass

    def on_unit_destroyed(self, unit: Unit):
        self.add_weight(-1, unit.position_tuple, 6.5, True)
        pass

    def on_unit_created(self, unit: Unit):
        self.add_weight(1, unit.position_tuple, 6.5, True)
        pass


class BuildingGrid(InfluenceGrid):
    def __init__(self, shape: tuple):
        super().__init__(shape, value_type=bool, init_value=True)

    def on_create(self, bot, map_data=None):
        for structure in bot.structures:
            self.set_weight(False, structure.position_tuple, structure.footprint_radius)
        pass

    def on_unit_destroyed(self, unit: Unit):
        self.set_weight(True, unit.position_tuple, unit.footprint_radius)
        pass

    def on_unit_created(self, unit: Unit):
        self.set_weight(False, unit.position_tuple, unit.footprint_radius)
        pass


class NaturalGrid(InfluenceGrid):
    def __init__(self, shape: tuple):
        super().__init__(shape, value_type=bool, init_value=True)

    def on_create(self, bot: BotAI, map_data=None):
        for unit in bot.mineral_field:
            pos = unit.position_tuple
            for x in range(round(pos[0] - 1), round(pos[0] + 1)):
                y = round(pos[1] - 0.5)
                self.grid[y][x] = False
        for unit in bot.vespene_geyser:
            pos = unit.position_tuple
            for x in range(round(pos[0] - 1.5), round(pos[0] + 1.5)):
                for y in range(round(pos[1] - 1.5), round(pos[1] + 1.5)):
                    self.grid[y][x] = False
        pass

    def on_unit_destroyed(self, unit: Unit):
        pos = unit.position_tuple
        if unit.type_id == UnitTypeId.VESPENEGEYSER:
            for x in range(round(pos[0] - 1.5), round(pos[0] + 1.5)):
                for y in range(round(pos[1] - 1.5), round(pos[1] + 1.5)):
                    self.grid[y][x] = True
        else:
            for x in range(round(pos[0] - 1), round(pos[0] + 1)):
                y = round(pos[1] - 0.5)
                self.grid[y][x] = True

    def on_unit_created(self, unit: Unit):
        pos = unit.position_tuple
        if unit.type_id == UnitTypeId.VESPENEGEYSER:
            for x in range(round(pos[0] - 1.5), round(pos[0] + 1.5)):
                for y in range(round(pos[1] - 1.5), round(pos[1] + 1.5)):
                    self.grid[y][x] = False
        else:
            for x in range(round(pos[0] - 1), round(pos[0] + 1)):
                y = round(pos[1] - 0.5)
                self.grid[y][x] = False

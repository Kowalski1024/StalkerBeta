from MapInfluence.influence_grid import InfluenceGrid

from MapAnalyzer.MapData import MapData
from sc2.bot_ai import BotAI
from sc2.unit import Unit


class PylonsGrid(InfluenceGrid):
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def on_create(self, bot: BotAI, map_data: MapData = None):
        point = bot.main_base_ramp.protoss_wall_pylon
        if point is not None:
            self.add_weight(2, point, 1)
        if map_data is not None:
            points = map_data.where_all(bot.townhalls[0].position)[0].buildables.points
            self.add_points_with_weight(10, points)
        pass

    def on_unit_destroyed(self, unit):
        pass

    def on_unit_created(self, unit):
        pass

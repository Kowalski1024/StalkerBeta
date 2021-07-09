from sc2.unit import Unit
from sc2.units import Units
from sc2.bot_ai import BotAI

from typing import Union, Set, Dict

from .region import Region


class PylonRegion(Region):
    def __init__(self, bot: BotAI):
        super().__init__(bot.game_info.map_size, init_value=20)
        self.pylons: Dict[int, tuple]
        self.bot = bot

    def update(self):
        pass

    def add_pylon(self, unit):
        pass


class PowerGrid(Region):
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def update(self):
        pass

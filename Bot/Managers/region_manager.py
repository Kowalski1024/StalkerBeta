
from .manager_base import ManagerBase
from sc2.game_info import GameInfo
from MapRegions import MapRegions


class RegionManager(ManagerBase):
    def __init__(self, bot):
        super().__init__()
        self._bot = bot
        self._game_info: GameInfo = bot.game_info
        self.regions = MapRegions(self._bot)

    async def update(self):
        pass

    async def post_update(self):
        pass

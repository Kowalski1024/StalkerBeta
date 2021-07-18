from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from sc2.bot_ai import BotAI


class ManagerBase:
    _bot: "BotAI"

    def __init__(self):
        self._debug = False

    @abstractmethod
    async def update(self):
        pass

    @abstractmethod
    async def post_update(self):
        pass

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from bot import BotBrain


class ManagerBase:
    _bot: "BotBrain"

    def __init__(self):
        self._debug = False

    @abstractmethod
    async def update(self):
        pass

    @abstractmethod
    async def post_update(self):
        pass

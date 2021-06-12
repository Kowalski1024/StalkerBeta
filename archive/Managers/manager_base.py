from typing import TYPE_CHECKING

from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from commander import Commander


class ManagerBase(ABC):
    bot: "Commander"

    def __init__(self):
        self._debug = False

    @abstractmethod
    async def update(self):
        pass

    @abstractmethod
    async def post_update(self):
        pass

    async def on_unit_destroyed(self, unit_tag):
        pass

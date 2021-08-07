from .manager_base import ManagerBase
from sc2.unit import Unit
from typing import Set, Dict


class TownhallsManager(ManagerBase):
    def __init__(self):
        super().__init__()
        self.townhalls_tags: Dict[int, Unit] = dict()
        self.build_workers: Dict[int, Unit] = dict()

    async def update(self):
        pass

    async def post_update(self):
        pass


class Townhall:
    def __init__(self):
        self.mineral_workers: Dict[int, Unit] = dict()
        self.gas_workers: Dict[int, Unit] = dict()

from .manager_base import ManagerBase
from BuildOrder.build_task import BuildTask
from BuildOrder.builder import Builder
from Managers.influence_gird_manager import InfluenceGridManager

from sc2.position import Point2
from sc2.client import Client
from sc2.constants import *

from typing import Union, List, Dict


class BuildOrderManager(ManagerBase):
    MAX_WORKERS = 80

    def __init__(self, bot, map_influence: InfluenceGridManager):
        super().__init__()
        self._bot = bot
        self._map_influence = map_influence
        self._order_list: List[BuildTask] = list()
        self._order_dict: Dict[UnitTypeId, BuildTask] = dict()
        self.safe_minerals = 0
        self.safe_vespene = 0

    def on_create(self):
        self.add_task(unit_id=UnitTypeId.PROBE, priority=1)
        task = self._order_dict[UnitTypeId.PROBE]
        task.priority = 2
        pass

    async def update(self):
        for task in self._order_list:
            if self.do_task(task):
                task.priority += 1
                return
        pass

    async def post_update(self):
        self.debug_draw()

        pass

    def add_task(self, unit_id: UnitTypeId, priority):
        task = BuildTask(unit_id=unit_id, priority=priority)
        self._order_dict[unit_id] = task
        self._order_list.append(task)
        self._order_list.sort()

    def do_task(self, task: BuildTask):
        unit_id = task.unit_id
        if unit_id == UnitTypeId.PROBE:
            return self.train_probe()

    def train_probe(self) -> bool:
        if(
            self.can_afford(UnitTypeId.PROBE)
        ):
            if(
                (len(self._bot.structures(UnitTypeId.NEXUS)) * 22) > len(self._bot.units(UnitTypeId.PROBE))
                and len(self._bot.units(UnitTypeId.PROBE)) < self.MAX_WORKERS
                and self._bot.townhalls.ready.idle
            ):
                nexus = self._bot.townhalls.ready.idle.random
                if nexus:
                    return nexus.train(UnitTypeId.PROBE)
        return False



    def can_afford(self, item_id: Union[UnitTypeId, UpgradeId, AbilityId], check_supply_cost: bool = True) -> bool:
        cost = self._bot.calculate_cost(item_id)
        if cost.minerals > self._bot.minerals - self.safe_minerals or cost.vespene > self._bot.vespene - self.safe_vespene:
            return False
        if check_supply_cost and isinstance(item_id, UnitTypeId):
            supply_cost = self._bot.calculate_supply_cost(item_id)
            if supply_cost and supply_cost > self._bot.supply_left:
                return False
        return True

    def debug_draw(self):
        msg = str()
        for task in self._order_list:
            priority = task.priority
            type_id = task.unit_id
            msg += f"P{priority}:{type_id}\n"

        client: Client = self._bot._client
        client.debug_text_screen(msg, Point2((0.01, 0.01)), None, 16)


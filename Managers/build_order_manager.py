from .manager_base import ManagerBase
from BuildOrder.build_task import BuildTask
from BuildOrder.builder import Builder
from Managers.influence_gird_manager import InfluenceGridManager

import sc2_math
from sc2.game_data import Cost
from sc2.position import Point2
from sc2.unit import Unit
from sc2.client import Client
from my_constants import *

from typing import Union, List, Dict


class BuildOrderManager(ManagerBase):
    MAX_WORKERS = 80

    def __init__(self, bot, map_influence: InfluenceGridManager):
        super().__init__()
        self._bot = bot
        self._map_influence = map_influence
        self._order_list: List[BuildTask] = list()
        self._order_dict: Dict[UnitTypeId, BuildTask] = dict()
        self.claimed_resources = Cost(0, 0)

    def on_create(self):
        self.add_task(unit_id=UnitTypeId.PROBE, priority=1, claim_resources=False)
        self.add_task(unit_id=UnitTypeId.PYLON, priority=3)
        pass

    async def update(self):
        self._order_list.sort(key=lambda t: t.priority)
        for task in self._order_list:
            if self.do_task(task) and task.unit_id not in TECH_STRUCTURES | BUILDING_STRUCTURES | DEF_STRUCTURES:
                self.reset_task(task)
            elif task.claim_resources:
                self.claimed_resources += self._bot.calculate_cost(task.unit_id)
        self.claimed_resources = Cost(0, 0)
        pass

    async def post_update(self):
        self.debug_draw()
        pylon_task = self._order_dict[UnitTypeId.PYLON]
        incoming_supply = self._bot.structures(UnitTypeId.PYLON).not_ready.amount * 8
        if self._bot.supply_left + incoming_supply != pylon_task.priority:
            pylon_task.priority = self._bot.supply_left + incoming_supply
        pass

    async def on_unit_created(self, unit: Unit):
        pass

    async def on_unit_destroyed(self, unit: Unit):
        pass

    async def on_building_construction_started(self, unit: Unit):
        unit_id = unit.type_id
        if (
                unit_id in self._order_dict
        ):
            task = self._order_dict[unit_id]
            self.reset_task(task)

        if unit_id == UnitTypeId.PYLON and UnitTypeId.GATEWAY not in self._order_dict:
            self.add_task(unit_id=UnitTypeId.GATEWAY, priority=1)
        pass

    async def on_building_construction_complete(self, unit: Unit):
        pass

    def add_task(self, unit_id: UnitTypeId, priority, claim_resources=True):
        task = BuildTask(unit_id=unit_id, priority=priority, claim_resources=claim_resources)
        self._order_dict[unit_id] = task
        self._order_list.append(task)

    def reset_task(self, task: BuildTask):
        self._order_list.remove(task)
        self._order_list.append(task)

    def remove_task(self, task: BuildTask):
        self._order_list.remove(task)
        self._order_dict.pop(task.unit_id)

    def do_task(self, task: BuildTask):
        unit_id = task.unit_id
        if unit_id == UnitTypeId.PROBE:
            return self.train_probe()
        elif unit_id == UnitTypeId.PYLON:
            return self.build_pylon()
        elif unit_id in TECH_STRUCTURES | BUILDING_STRUCTURES:
            return self.build_structure(unit_id)

    def train_probe(self) -> bool:
        if (
                self.can_afford(UnitTypeId.PROBE)
        ):
            if (
                    (len(self._bot.structures(UnitTypeId.NEXUS)) * 22) > len(self._bot.units(UnitTypeId.PROBE))
                    and len(self._bot.units(UnitTypeId.PROBE)) < self.MAX_WORKERS
                    and self._bot.townhalls.ready.idle
            ):
                nexus = self._bot.townhalls.ready.idle.random
                if nexus:
                    return nexus.train(UnitTypeId.PROBE)
        return False

    def build_pylon(self) -> bool:
        def build(target_pos):
            worker = self._bot.select_build_worker(target_pos)
            return worker.build(UnitTypeId.PYLON, target_pos)

        if (
                self.can_afford(UnitTypeId.PYLON)
        ):
            reg = self._map_influence.get_pylon_grid()
            return build(sc2_math.find_building_position(reg.grid, 2))
        return False

    # TODO: get to destination before tracking building construction complete
    def build_structure(self, unit_id: UnitTypeId) -> bool:
        if self.can_afford(unit_id):
            reg = self._map_influence.get_building_grid()
            pos = sc2_math.find_building_position(reg.grid, 3, 9)
            if pos is None or not self._bot.can_place_single(unit_id, pos):
                return False
            worker = self._bot.select_build_worker(pos)
            return worker.build(unit_id, pos)
        return False
        # if task.target_position is None:
        #     reg = self._map_influence.get_building_grid()
        #     pos = sc2_math.find_building_position(reg.grid, 3)
        #     if pos is None:
        #         return False
        #     if self._bot.can_place_single(UnitTypeId.GATEWAY, pos):
        #         task.target_position = pos
        #     else:
        #         return False
        # worker = self._bot.select_build_worker(task.target_position)
        # if self.can_afford(UnitTypeId.GATEWAY):
        #     return worker.build(UnitTypeId.GATEWAY, task.target_position)
        # elif (
        #         self._bot.structures(UnitTypeId.PYLON).ready.amount == 0
        #         and self._bot.structures(UnitTypeId.PYLON).not_ready
        # ):
        #     track_unit = self._bot.structures(UnitTypeId.PYLON).not_ready.random
        #     travel_time = round(worker.distance_to(task.target_position) / worker.movement_speed, 2)
        #     building_time = self.get_unit_info(UnitTypeId.PYLON, "build_time")
        #     building_end_time = round((building_time - building_time * track_unit.build_progress) / 22.4 + 1.5, 2)
        #     if (
        #             travel_time >= building_end_time
        #             and track_unit.build_progress < 1
        #     ):
        #         worker.move(task.target_position)
        # return False

    def can_afford(self, item_id: Union[UnitTypeId, UpgradeId, AbilityId], check_supply_cost: bool = True) -> bool:
        cost = self._bot.calculate_cost(item_id)
        if (
                cost.minerals > self._bot.minerals - self.claimed_resources.minerals
                or cost.vespene > self._bot.vespene - self.claimed_resources.vespene
        ):
            return False
        if check_supply_cost and isinstance(item_id, UnitTypeId):
            supply_cost = self._bot.calculate_supply_cost(item_id)
            if supply_cost and supply_cost > self._bot.supply_left:
                return False
        return True

    def get_unit_info(self, unit, field="build_time"):
        """
        get various unit data
        usage: getUnitInfo(ROACH, "mineral_cost")

        :param unit:
        :param field:
        """
        assert isinstance(unit, (Unit, UnitTypeId))
        if isinstance(unit, Unit):
            unit = unit._type_data._proto
        else:
            unit = self._bot._game_data.units[unit.value]._proto
        if hasattr(unit, field):
            return getattr(unit, field)

    def debug_draw(self):
        msg = str()
        for task in self._order_list:
            priority = task.priority
            type_id = task.unit_id
            msg += f"P{priority}:{type_id}\n"

        client: Client = self._bot._client
        client.debug_text_screen(msg, Point2((0.01, 0.01)), None, 16)

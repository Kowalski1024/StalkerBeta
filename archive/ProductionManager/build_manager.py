from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING

from sc2.constants import *
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ProductionManager.builder import Builder

if TYPE_CHECKING:
    from commander import Commander


class BuildManager(Builder):
    def __init__(self, bot: "Commander"):
        super().__init__(bot)
        pass

    async def builder_manager(self):
        if self.bot.structures(UnitTypeId.ASSIMILATOR).amount < self.structures_dict["assimilators"]:
            await self.build_assimilators()

        if self.bot.iteration % 2 == 0:
            if self.structures_dict["assimilators"] == 0 and self.bot.structures(UnitTypeId.GATEWAY).exists:
                self.structures_dict["assimilators"] = 1
            if (
                    self.bot.supply_left <= self.supply_need
                    and self.bot.can_afford(UnitTypeId.PYLON)
                    and not self.bot.already_pending(UnitTypeId.PYLON)
                    and not self.bot.supply_cap == 200
            ):
                await self.build_pylon()

            await self.build_gateway()
            if (
                    self.structures_dict["cybernetics_core"]
            ):
                await self.build_cybernetics_core()

            if self.bot.structures(UnitTypeId.CYBERNETICSCORE).ready and self.structures_dict["gateway"] < 2:
                self.structures_dict["gateway"] = 1
                self.structures_dict["assimilators"] = 2

            # # research warpgate
            # if (
            #     not self.structures(UnitTypeId.WARPGATE)
            #     and self.structures(UnitTypeId.CYBERNETICSCORE).ready
            #     and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0
            #     and self.can_afford(AbilityId.RESEARCH_WARPGATE)
            # ):
            #     cyber_core = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            #     cyber_core.research(UpgradeId.WARPGATERESEARCH)

            await self.train_probe()
            if self.bot.structures(UnitTypeId.GATEWAY).exists:
                await self.train_stalker()
        pass

    def build_fast_as_possible(self, unit_type, track_unit, target_location):
        """
        :param unit_type:
        :param track_unit:
        :param target_location:
        """
        workers: Units = self.bot.units(UnitTypeId.PROBE)
        if workers:
            worker: Unit = workers.closest_to(target_location)
            travel_time = round(worker.distance_to(target_location) / worker.movement_speed, 2)
            building_time = self.bot.get_unit_info(track_unit.type_id, "build_time")
            building_end_time = round((building_time - building_time * track_unit.build_progress) / 22.4 + 1.5, 2)
            if (
                    travel_time >= building_end_time
                    and track_unit.build_progress < 1
                    and self.bot.iteration % 2 == 0
            ):
                worker.move(target_location)
                self.save_minerals = 125
            elif track_unit.build_progress == 1:
                if self.bot.do(worker.build(unit_type, target_location)):
                    self.save_minerals = 0

    async def build_ramp_wall(self):
        # get position of wall buildings
        try:
            pylon_wall_position = self.bot.main_base_ramp.protoss_wall_pylon
            buildings_wall_position: Set[Point2] = self.bot.main_base_ramp.protoss_wall_buildings
        except:
            self.ramp_exists = False
            return
        pylon_wall_exists = False
        gateway_wall_exists = False

        # checking if wall buildings exists
        if self.bot.structures(UnitTypeId.PYLON).exists:
            pylon = self.bot.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            if pylon.position == pylon_wall_position:
                pylon_wall_exists = True
        for d in buildings_wall_position:
            if self.bot.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE, UnitTypeId.CYBERNETICSCORE}).exists:
                building = self.bot.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).closest_to(d)
                if building.position == d:
                    if building.type_id == UnitTypeId.GATEWAY:
                        gateway_wall_exists = True

        if pylon_wall_exists and gateway_wall_exists and self.bot.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return

        # build pylon wall if not exists
        if (
                self.bot.can_afford(UnitTypeId.PYLON)
                and self.bot.already_pending(UnitTypeId.PYLON) == 0
                and not pylon_wall_exists
        ):
            worker = self.bot.select_build_worker(pylon_wall_position)
            if worker:
                self.bot.do(worker.build(UnitTypeId.PYLON, pylon_wall_position))

        # first gateway
        if (
                pylon_wall_exists
                and not gateway_wall_exists
        ):
            target_building_location: Point2 = buildings_wall_position.pop()
            pylon = self.bot.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            self.build_fast_as_possible(UnitTypeId.GATEWAY, pylon, target_building_location)

        # clear gateway position
        gateway_wall_position: Units = self.bot.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE})
        if gateway_wall_position:
            buildings_wall_position: Set[Point2] = {
                d for d in buildings_wall_position if gateway_wall_position.closest_distance_to(d) > 1
            }

        # Build cybernetics core
        if (
                self.bot.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).exists
                and self.bot.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.bot.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            gateway_wall = gateway_wall_position.random
            target_building_location: Point2 = buildings_wall_position.pop()
            self.build_fast_as_possible(UnitTypeId.CYBERNETICSCORE, gateway_wall, target_building_location)
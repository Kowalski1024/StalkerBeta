from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from sc2.constants import *

import numpy as np

if TYPE_CHECKING:
    from commander import Commander


class Builder:
    def __init__(self, bot: "Commander"):
        self.bot = bot

        # Const
        self.MAX_WORKERS = 50

        # Bot data
        self.save_minerals = 0
        self.supply_need = 3

        self.structures_dict = {
            "nexuses": 0,
            "pylons": 0,
            "assimilators": 0,
            "gateway": 0,
            "stargate": 0,
            "robotics_facility": 0,
            "forge": 0,
            "cybernetics_core": False,
            "robotics_bay": False,
            "twilight_council": False,
            "templar_archives": False,
            "dark_shrine": False,
            "fleet_beacon": False
        }
        pass

    async def train_probe(self):
        if (
                (len(self.bot.structures(UnitTypeId.NEXUS)) * 22) > len(self.bot.units(UnitTypeId.PROBE))
                and len(self.bot.units(UnitTypeId.PROBE)) < self.MAX_WORKERS
                and self.bot.minerals - self.save_minerals >= 50
        ):
            for nexus in self.bot.structures(UnitTypeId.NEXUS).ready.idle:
                if self.bot.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)
        pass

    async def train_stalker(self):
        if (
                self.bot.structures(UnitTypeId.CYBERNETICSCORE).ready
                and self.bot.structures.of_type({UnitTypeId.WARPGATE, UnitTypeId.GATEWAY})
                and self.bot.can_afford(UnitTypeId.STALKER)

        ):
            await self.bot.train_unit(UnitTypeId.STALKER)
        pass

    async def build_gateway(self):
        if (
                self.bot.can_afford(UnitTypeId.GATEWAY)
                and self.bot.structures(UnitTypeId.PYLON).ready
                and not self.bot.already_pending(UnitTypeId.GATEWAY)
                and self.bot.structures(UnitTypeId.GATEWAY).amount < self.structures_dict["gateway"]
        ):
            pylon = self.bot.structures(UnitTypeId.PYLON).ready.random
            await self.bot.build(UnitTypeId.GATEWAY, near=pylon)
        pass

    async def build_stargate(self):
        pass

    async def build_robotics_facility(self):
        pass

    async def build_pylon(self):
        def build(target_pos):
            worker = self.bot.select_build_worker(target_pos)
            worker.build(UnitTypeId.PYLON, target_pos)

        if self.bot.can_afford(UnitTypeId.PYLON):
            nexuses = self.bot.structures(UnitTypeId.NEXUS)
            pylon_region = self.bot.regions.placement - 32 * self.bot.regions.building_map - self.bot.regions.pylon_grid
            structures = self.bot.structures.exclude_type(
                {UnitTypeId.NEXUS, UnitTypeId.PYLON, UnitTypeId.ASSIMILATOR}).filter(
                lambda unit: self.bot.regions.pylon_amount(unit) <= 2)
            for structure in structures:
                val = 2 - self.bot.regions.pylon_amount(structure)
                val = np.sign(val)*(val ** 2)
                self.bot.regions.pylon_grid_update(structure, val, grid=pylon_region)
            for nexus in nexuses:
                pylon = self.bot.regions.pylon_amount(nexus, dis=7)
                if not pylon:
                    pos = self.bot.regions.find_best_position(matrix=np.copy(pylon_region), dim=7, center=nexus, size=2)
                    build(pos)
                    return
            place = self.bot.regions.max_sub_matrix(matrix_in=pylon_region, size=2, min_value=100)[0]
            pos = Point2((place[3], place[1]))
            build(pos)
        pass

    async def build_cybernetics_core(self):
        if(
                self.bot.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.bot.structures(UnitTypeId.GATEWAY).exists
                and self.bot.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.bot.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            pylon = self.bot.structures(UnitTypeId.PYLON).ready.random
            await self.bot.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
        pass

    async def build_assimilators(self):
        for nexus in self.bot.townhalls.ready:
            vgs = self.bot.vespene_geyser.closer_than(15, nexus)
            vg = vgs.random
            if (
                    not self.bot.already_pending(UnitTypeId.ASSIMILATOR)
                    or self.bot.structures(UnitTypeId.ASSIMILATOR).amount + 1 < self.structures_dict["assimilators"]
            ):
                worker = self.bot.select_build_worker(vg.position)
                if not self.bot.gas_buildings or not self.bot.gas_buildings.closer_than(1, vg):
                    self.bot.do(worker.build(UnitTypeId.ASSIMILATOR, vg))
                    worker.stop(queue=True)
        pass



from archive.bot_addons import MyBotAddons
from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.constants import *
from loguru import logger
import numpy as np
import math
import random


class BuilderManager(MyBotAddons):
    def __init__(self):
        super().__init__()
        self.MAX_WORKERS = 50

    async def on_step(self, iteration: int):
        pass

    async def builder_manager(self):
        if self.structures(UnitTypeId.ASSIMILATOR).amount < self.structures_dict["assimilators"]:
            await self.build_assimilators()

        if self.iteration % 2 == 0:
            if self.structures_dict["assimilators"] == 0 and self.structures(UnitTypeId.GATEWAY).exists:
                self.structures_dict["assimilators"] = 1
            if (
                    self.supply_left <= self.supply_need
                    and self.can_afford(UnitTypeId.PYLON)
                    and not self.already_pending(UnitTypeId.PYLON)
                    and not self.supply_cap == 200
            ):
                await self.build_pylon()

            await self.build_gateway()
            if (
                    self.structures_dict["cybernetics_core"]
            ):
                await self.build_cybernetics_core()

            if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.structures_dict["gateway"] < 2:
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
            if self.structures(UnitTypeId.GATEWAY).exists:
                await self.train_stalker()
        pass

    async def train_probe(self):
        if (
                (len(self.structures(UnitTypeId.NEXUS)) * 22) > len(self.units(UnitTypeId.PROBE))
                and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS
                and self.minerals - self.save_minerals >= 50
        ):
            for nexus in self.structures(UnitTypeId.NEXUS).ready.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)
        pass

    async def train_stalker(self):
        if (
                self.structures(UnitTypeId.CYBERNETICSCORE).ready
                and self.structures.of_type({UnitTypeId.WARPGATE, UnitTypeId.GATEWAY})
                and self.can_afford(UnitTypeId.STALKER)

        ):
            await self.train_unit(UnitTypeId.STALKER)
        pass

    async def build_gateway(self):
        if (
                self.can_afford(UnitTypeId.GATEWAY)
                and self.structures(UnitTypeId.PYLON).ready
                and not self.already_pending(UnitTypeId.GATEWAY)
                and self.structures(UnitTypeId.GATEWAY).amount < self.structures_dict["gateway"]
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)
        pass

    async def build_stargate(self):
        pass

    async def build_robotics_facility(self):
        pass

    async def build_pylon(self):
        def build(target_pos):
            worker = self.select_build_worker(target_pos)
            worker.build(UnitTypeId.PYLON, target_pos)

        if self.can_afford(UnitTypeId.PYLON):
            nexuses = self.structures(UnitTypeId.NEXUS)
            pylon_region = self.regions.placement - 32 * self.regions.building_map - self.regions.pylon_grid
            structures = self.structures.exclude_type(
                {UnitTypeId.NEXUS, UnitTypeId.PYLON, UnitTypeId.ASSIMILATOR}).filter(
                lambda unit: self.regions.pylon_amount(unit) <= 2)
            for structure in structures:
                val = 2 - self.regions.pylon_amount(structure)
                val = np.sign(val)*(val ** 2)
                self.regions.pylon_grid_update(structure, val, grid=pylon_region)
            for nexus in nexuses:
                pylon = self.regions.pylon_amount(nexus, dis=7)
                if not pylon:
                    pos = self.regions.find_best_position(matrix=np.copy(pylon_region), dim=7, center=nexus, size=2)
                    build(pos)
                    return
            place = self.regions.max_sub_matrix(matrix_in=pylon_region, size=2, min_value=100)[0]
            pos = Point2((place[2], place[1]))
            build(pos)
        pass

    async def build_cybernetics_core(self):
        if(
                self.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.structures(UnitTypeId.GATEWAY).exists
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
        pass

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15, nexus)
            vg = vgs.random
            if (
                    not self.already_pending(UnitTypeId.ASSIMILATOR)
                    or self.structures(UnitTypeId.ASSIMILATOR).amount + 1 < self.structures_dict["assimilators"]
            ):
                worker = self.select_build_worker(vg.position)
                if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                    self.do(worker.build(UnitTypeId.ASSIMILATOR, vg))
                    worker.stop(queue=True)
        pass

    def build_fast_as_possible(self, unit_type, track_unit, target_location):
        """
        :param unit_type: 
        :param track_unit: 
        :param target_location: 
        """
        workers: Units = self.units(UnitTypeId.PROBE)
        if workers:
            worker: Unit = workers.closest_to(target_location)
            travel_time = round(worker.distance_to(target_location) / worker.movement_speed, 2)
            building_time = self.get_unit_info(track_unit.type_id, "build_time")
            building_end_time = round((building_time - building_time * track_unit.build_progress) / 22.4 + 1.5, 2)
            if (
                    travel_time >= building_end_time
                    and track_unit.build_progress < 1
                    and self.iteration % 2 == 0
            ):
                worker.move(target_location)
                self.save_minerals = 125
            elif track_unit.build_progress == 1:
                if self.do(worker.build(unit_type, target_location)):
                    self.save_minerals = 0

    async def build_ramp_wall(self):
        # get position of wall buildings
        try:
            pylon_wall_position = self.main_base_ramp.protoss_wall_pylon
            buildings_wall_position: Set[Point2] = self.main_base_ramp.protoss_wall_buildings
        except:
            self.ramp_exists = False
            return
        pylon_wall_exists = False
        gateway_wall_exists = False

        # checking if wall buildings exists
        if self.structures(UnitTypeId.PYLON).exists:
            pylon = self.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            if pylon.position == pylon_wall_position:
                pylon_wall_exists = True
        for d in buildings_wall_position:
            if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE, UnitTypeId.CYBERNETICSCORE}).exists:
                building = self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).closest_to(d)
                if building.position == d:
                    if building.type_id == UnitTypeId.GATEWAY:
                        gateway_wall_exists = True

        if pylon_wall_exists and gateway_wall_exists and self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return

        # build pylon wall if not exists
        if (
                self.can_afford(UnitTypeId.PYLON)
                and self.already_pending(UnitTypeId.PYLON) == 0
                and not pylon_wall_exists
        ):
            worker = self.select_build_worker(pylon_wall_position)
            if worker:
                self.do(worker.build(UnitTypeId.PYLON, pylon_wall_position))

        # first gateway
        if (
                pylon_wall_exists
                and not gateway_wall_exists
        ):
            target_building_location: Point2 = buildings_wall_position.pop()
            pylon = self.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            self.build_fast_as_possible(UnitTypeId.GATEWAY, pylon, target_building_location)

        # clear gateway position
        gateway_wall_position: Units = self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE})
        if gateway_wall_position:
            buildings_wall_position: Set[Point2] = {
                d for d in buildings_wall_position if gateway_wall_position.closest_distance_to(d) > 1
            }

        # Build cybernetics core
        if (
                self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).exists
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            gateway_wall = gateway_wall_position.random
            target_building_location: Point2 = buildings_wall_position.pop()
            self.build_fast_as_possible(UnitTypeId.CYBERNETICSCORE, gateway_wall, target_building_location)


class Regions:
    def __init__(self, bot: MyBotAddons):
        self.bot = bot

        self.reg_dic = {
            "resource": 40,
            "resource_region": 70,
            "placement_grid": 128
        }

        # regions
        self.main_base = list()
        self.enemy_base = list()
        self.enemy_unseen_region = set(self.enemy_base)

        # main placements grid
        self.placement = bot.game_info.placement_grid.copy().data_numpy
        self.visibility = self.bot.state.visibility.data_numpy
        self.shape = self.placement.shape

        # other placements grid
        self.building_map = np.full(self.shape, False, dtype=bool)
        self.building_grid = np.full(self.shape, 0)
        self.pylon_grid = self.placement.copy() * 10
        self.on_create()

    def on_create(self):
        self.placement[self.placement == 1] = self.reg_dic["placement_grid"]
        # add mineral field  and vespene geyser to placement grid
        for resource in self.bot.resources:
            pos = resource.position
            if resource in self.bot.vespene_geyser:
                for x in range(round(pos[0] - 1.5), round(pos[0] + 1.5)):
                    for y in range(round(pos[1] - 1.5), round(pos[1] + 1.5)):
                        self.placement[y][x] = self.reg_dic["resource"]
            else:
                for x in range(round(pos[0] - 1), round(pos[0] + 1)):
                    self.placement[round(pos[1] - 0.5)][x] = self.reg_dic["resource"]

        # minerals region on main
        resources = self.bot.resources.filter(lambda unit: unit.distance_to(self.bot.townhalls[0]) <= 10)
        self.facing_region(self.placement, resources, self.bot.townhalls[0], dis=-2, num=1, add=True)
        self.facing_region(self.placement, resources, self.bot.townhalls[0], dis=4, num=self.reg_dic["resource_region"])
        self.facing_region(self.placement, self.bot.main_base_ramp.protoss_wall_warpin, self.bot.townhalls[0], dis=2,
                           num=70, offset_angle=math.pi / 6)
        pass

    def facing_region(self, matrix, my_units, to_position, dis: int = 4, num: int = 128, add: bool = False,
                      offset_angle: float = math.pi / 3):
        """
        :param matrix:
        :param my_units:
        :param to_position:
        :param dis:
        :param num:
        :param add:
        :param offset_angle:
        """
        def region(unit_position):
            angle = math.atan2(to_position[1] - unit_position[1], to_position[0] - unit_position[0])
            if dis < 0:
                angle -= math.pi
            if angle < 0:
                angle += math.pi * 2
            vector_one = (math.cos(angle - offset_angle), math.sin(angle - offset_angle))
            vector_two = (math.cos(angle + offset_angle), math.sin(angle + offset_angle))
            point_list = list()
            for x in np.arange(0.2, abs(dis), 0.2):
                for y in np.arange(0.2, abs(dis), 0.2):
                    vector_sum = (x * (vector_one[0] + vector_two[0]), y * (vector_one[1] + vector_two[1]))
                    y_pos, x_pos = round(unit_position[1] + vector_sum[1]), round(unit_position[0] + vector_sum[0])
                    if 100 < matrix[y_pos][x_pos] != 0:
                        if (y_pos, x_pos) not in point_list:
                            point_list.append(tuple((y_pos, x_pos)))
                            if add:
                                matrix[y_pos][x_pos] += num
                            else:
                                matrix[y_pos][x_pos] = num

        if matrix.shape != self.shape:
            logger.debug("Wrong shape")
            return

        if isinstance(to_position, Unit):
            to_position = to_position.position_tuple

        if isinstance(my_units, Point2):
            region(my_units)
        elif isinstance(my_units, Unit):
            region(my_units.position_tuple)
        else:
            for u in my_units:
                region(u.position_tuple)
        pass

    def update_structures(self, structure: Unit = None):
        def add_structure(unit: Unit):
            radius = unit.footprint_radius
            pos = unit.position
            for x in range(round(pos[0] - radius), round(pos[0] + radius)):
                for y in range(round(pos[1] - radius), round(pos[1] + radius)):
                    self.building_map[y][x] = True

        if structure is None:
            for s in self.bot.structures + self.bot.enemy_structures:
                add_structure(s)
        else:
            add_structure(structure)
        pass

    def pylon_amount(self, my_unit: Unit, dis=6.5):
        pylons = self.bot.structures(UnitTypeId.PYLON).filter(
            lambda unit: unit.distance_to_squared(my_unit) <= dis ** 2)
        return pylons.amount
        pass

    def pylon_grid_update(self, unit: Unit, val: int = 1, grid=None):
        """
        :param unit:
        :param val:
        :param grid:
        """
        if grid is None:
            grid = self.pylon_grid
        elif grid.shape != self.shape:
            logger.debug("Wrong shape")
            return
        radius = 6
        pos = unit.position
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x ** 2 + y ** 2 <= radius ** 2:
                    grid[int(x + pos[1]), int(y + pos[0])] += val
        pass

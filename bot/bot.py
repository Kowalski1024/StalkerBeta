import numpy as np

import sc2
from sc2.constants import *
from sc2 import Race, BotAI
from sc2.player import Bot, Computer, Difficulty
from sc2.unit import Unit
from sc2.position import Point2, Point3
from MapAnalyzer.MapData import MapData
import matplotlib.pyplot as plt
from MapAnalyzer.constructs import RawChoke
from collections import deque
from typing import Union, Tuple
import math
from itertools import chain
from more_itertools import pairwise
from MapAnalyzer.constructs import MDRamp
from Bot.Managers.MapInfluence.grid_types import GridTypes
from InfluenceGrid import GridData
from MapRegions import MapRegions
import time

from Managers.townhalls_manager import Townhall
from Managers.influence_gird_manager import InfluenceGridManager
from Managers.region_manager import RegionManager
from Managers.build_order_manager import BuildOrderManager
from Managers.BuildOrder.Wallin import WallBuilder

from MapRegions import ConnectivitySide

GAME_STEP = 8
DATA_PATH = "Data/"


class BotBrain(BotAI):
    map_data: MapData
    townhall: Townhall
    region_data: MapRegions
    grid_data: GridData
    builder_manager: BuildOrderManager

    def __init__(self):
        super().__init__()
        self.iteration = 0

        self.units_dict: Dict[int, Unit] = dict()
        self.test_list = list()

    async def on_unit_destroyed(self, unit_tag):
        pass

    async def on_building_construction_started(self, unit: Unit):
        pass

    async def on_building_construction_complete(self, unit: Unit):
        pass

    async def on_unit_created(self, unit: Unit):
        if unit.type_id == UnitTypeId.PROBE:
            self.townhall.register_worker(unit)
        pass

    async def on_before_start(self):
        self.units_dict = {unit.tag: unit for unit in self.units}
        fields = self.mineral_field.filter(lambda unit: unit.distance_to(self.townhalls[0]) <= 10)
        self.townhall = Townhall(fields, self, self.townhalls[0])
        pass

    async def on_start(self):
        self.client.game_step = 4
        self.region_data = MapRegions(self)
        self.grid_data = GridData(self, self.region_data)
        self.builder_manager = BuildOrderManager(self, self.grid_data)
        self.builder_manager.on_create()
        for conn in self.region_data.regions[2].connectivity_dict.values():
            for c in conn:
                wall = WallBuilder(c.points, self.game_info)
                x = wall.wall_search([3, 1, 3, 2, 3])
                print(x)
                self.test_list.extend(x)

        print("Game started")

    async def on_step(self, iteration):
        for p, r in self.test_list:
            if r == 2:
                p += Point2((-0.5, -0.5))
            h = self.get_terrain_z_height(p)
            self.client.debug_sphere_out(Point3((p[0]-1.5, p[1]-1.5, h+0.1)), r/2)
        # await self.builder_manager.update()
        self.iteration = iteration
        self.region_data.draw_connectivity_lines(include_points=True)
        for unit in self.units:
            self.units_dict[unit.tag] = unit
        # await self.builder_manager.post_update()

    def on_end(self, result):
        print("Game ended.", result)


def main():
    # BlackburnAIE RomanticideAIE 2000AtmospheresAIE LightshadeAIE JagannathaAIE
    # GlitteringAshesAIE, HardwireAIE, CuriousMindsAIE, BerlingradAIE
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("JagannathaAIE"), [
        Bot(Race.Protoss, BotBrain()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=True, disable_fog=True, random_seed=2)


if __name__ == '__main__':
    main()

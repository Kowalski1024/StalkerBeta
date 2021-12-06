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

from Managers.townhalls_manager import Townhall
from Managers.influence_gird_manager import InfluenceGridManager
from Managers.region_manager import RegionManager

from MapRegions import ConnectivitySide

GAME_STEP = 8
DATA_PATH = "Data/"


class BotBrain(BotAI):
    map_data: MapData
    townhall: Townhall
    influence_manager: InfluenceGridManager
    region_manager: RegionManager

    def __init__(self):
        super().__init__()
        self.iteration = 0

        self.units_dict: Dict[int, Unit] = dict()
        self.test_list = list()
        self.chokes = list()

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
        self.region_manager = RegionManager(self)
        print("Game started")

    async def on_step(self, iteration):
        self.iteration = iteration
        self.region_manager.regions.draw_connectivity_lines()
        for unit in self.units:
            self.units_dict[unit.tag] = unit
        pass

    def on_end(self, result):
        print("Game ended.", result)


# def save_obj(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
#
#
# def open_obj(filename):
#     with open(filename, 'rb') as file:
#         obj = pickle.load(file)
#     return obj


def main():
    # BlackburnAIE RomanticideAIE 2000AtmospheresAIE LightshadeAIE JagannathaAIE
    # EternalEmpire506
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("RomanticideAIE"), [
        Bot(Race.Protoss, BotBrain()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=True, disable_fog=True, random_seed=2)


if __name__ == '__main__':
    main()

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
from typing import Union
import math
from itertools import chain
from more_itertools import pairwise
from MapAnalyzer.constructs import MDRamp

from Managers.townhalls_manager import Townhall
from Managers.influence_gird_manager import InfluenceGridManager
from Managers.region_manager import RegionManager

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
        self.map_data = MapData(self)
        self.region_manager = RegionManager(self)
        self.region_manager.draw_polygons()
        self.draw_ramp_points()
        # x, y = self.townhalls.first.position
        # print(_map[int(y)][int(x)]) #191
        # x, y = self.get_next_expansion()
        # print(_map[int(y)][int(x)])
        # self.depth_points(self.townhalls.first)
        print("Game started")

    def print(self):
        for point in self.test_list:
            p = Point3((*point, self.townhalls.first.position3d[2]))
            self.client.debug_sphere_out(p, r=0.5)

    def height_at(self, p: Point2):
        return self.game_info.terrain_height.data_numpy[p.y][p.x]
        pass

    def lower(self, points) -> Set[Point2]:
        current_min = 10000
        result = set()
        for p in points:
            height = self.height_at(p)
            if height < current_min:
                print(height)
                current_min = height
                result = {p}
            elif height == current_min:
                result.add(p)
        return result

    def draw_ramp_points(self):
        for ramp in self.game_info.map_ramps:
            height_points = sorted(list({self.get_terrain_z_height(point) for point in ramp.points}))
            top_height = height_points[-3]
            bottom_height = height_points[2]
            upper = []
            down = []
            for p in ramp.points:
                h = self.get_terrain_z_height(p)
                pos = Point3((p.x, p.y, h+1))
                if h == top_height:
                    upper.append(p)
                    self.client.debug_box2_out(pos + Point2((0.5, 0.5)), half_vertex_length=0.25,
                                               color=Point3((0, 255, 0)))
                elif h == bottom_height:
                    down.append(p)
                    self.client.debug_box2_out(pos + Point2((0.5, 0.5)), half_vertex_length=0.25,
                                               color=Point3((255, 0, 0)))
            upper_center = Point2.center(upper)
            down_center = Point2.center(down)
            pos = Point3((*upper_center, self.get_terrain_z_height(upper_center)+1))
            self.client.debug_box2_out(pos + Point2((0.5, 0.5)), half_vertex_length=0.25,
                                       color=Point3((0, 0, 255)))
            pos = Point3((*down_center, self.get_terrain_z_height(down_center)))
            self.client.debug_box2_out(pos + Point2((0.5, 0.5)), half_vertex_length=0.25,
                                       color=Point3((0, 0, 255)))

    async def on_step(self, iteration):
        self.iteration = iteration
        # for ramp in self.game_info.map_ramps:
        #     for point in ramp.points:
        #         h = self.get_terrain_z_height(point)
        #         self.client.debug_text_world(text=str(h),
        #                                      pos=Point3((*point, self.townhalls.first.position3d.z)) + Point2(
        #                                          (0.5, 0.5)))
        self.draw_ramp_points()
        # self.region_manager.draw_chokes()
        # self.print()
        for unit in self.units:
            self.units_dict[unit.tag] = unit
        # await self.townhall.speed_mining()
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
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("BlackburnAIE"), [
        Bot(Race.Protoss, BotBrain()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=True, disable_fog=True)


if __name__ == '__main__':
    main()

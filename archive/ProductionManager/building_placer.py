from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING

from sc2 import BotAI
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.units import Units
from sc2.constants import *
from typing import List
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random

if TYPE_CHECKING:
    from commander import Commander


class Regions:
    def __init__(self, bot: "Commander"):
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
        pass

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
        self.facing_region(self.placement, resources, self.bot.townhalls[0], dis=3, num=self.reg_dic["resource_region"])
        self.facing_region(self.placement, self.bot.main_base_ramp.protoss_wall_warpin, self.bot.townhalls[0], dis=2,
                           offset_angle=math.pi/6,
                           num=70)
        pass

    def find_best_position(self, matrix, dim: float, center, size: int = 1, min_value: int = 128):
        """
        :param matrix:
        :param dim:
        :param center:
        :param size:
        :param min_value:
        :return:
        """
        sliced_matrix, start_pos = self.slice_region(matrix, dim, center)
        pos = self.max_sub_matrix(sliced_matrix, size, min_value)[0]
        return Point2((start_pos[1] + pos[3], start_pos[0] + pos[1]))
        pass

    def slice_region(self, matrix, dim: float, center_position):
        """
        :param matrix:
        :param dim:
        :param center_position:
        :return [new_matrix, (x, y)]:
        """
        if isinstance(center_position, Unit):
            center_position = center_position.position
        if center_position[0] - int(center_position[0]) == 0.5 and dim % 2 == 0:
            dim += 1
        elif center_position[0] - int(center_position[0]) == 0 and dim % 2 == 1:
            dim += 1

        x, y = (round(center_position[1] - dim / 2), round(center_position[0] - dim / 2))
        new_matrix = matrix[x:(x + dim), y:(y + dim)]
        return [new_matrix, (x, y)]
        pass

    def max_sub_matrix(self, matrix_in: np.array, size: int = 1, min_value=128):
        """
        :param matrix_in:
        :param size:
        :param min_value:
        :return [(y_start, y_end, x_start, x_end), max_sum, matrix_out]:
        """
        max_sum = 0
        pos = [0, 0, 0, 0]
        pos_list = list()
        matrix_size = matrix_in.shape
        matrix = np.full(matrix_size, -min_value + 1)
        matrix += matrix_in
        if size == 1:
            # Kadane algorithm
            for left in range(matrix_size[0]):
                for right in range(left, matrix_size[0]):
                    sub_matrix = matrix[:matrix_size[1] + 1, left:right + 1]
                    side_array = np.sum(sub_matrix, axis=1)
                    sum_arr = 0
                    start = 0
                    end = 0
                    while end < side_array.shape[0]:
                        sum_arr += side_array[end]
                        if sum_arr < 0:
                            sum_arr = 0
                            start = end + 1
                        elif sum_arr > max_sum:
                            max_sum = sum_arr
                            pos = [start, end, left, right]
                        elif (sum_arr == max_sum
                              and size > 1
                              and (pos[1] - pos[0]) * (pos[3] - pos[2]) < (right - left) * (end - start)
                        ):
                            max_sum = sum_arr
                            pos = [start, end, left, right]
                        end += 1
        else:
            size -= 1
            for x in range(matrix_size[0] - size):
                for y in range(matrix_size[1] - size):
                    sub_matrix = matrix[x:(x + size + 1), y:(y + size + 1)]
                    sum_arr = np.sum(sub_matrix)
                    if sum_arr >= 0:
                        if sum_arr > max_sum:
                            pos_list.clear()
                            max_sum = sum_arr
                            pos_list.append([x, x + size, y, y + size])
                        elif sum_arr == max_sum:
                            pos_list.append([x, x + size, y, y + size])
        if pos_list:
            pos = random.choice(pos_list)
        matrix_out = matrix_in[pos[0]:pos[1] + 1, pos[2]:pos[3] + 1]
        return [pos, max_sum, matrix_out]
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
                    vector_sum = (x*vector_one[0] + y*vector_two[0], x*vector_one[1] + y*vector_two[1])
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

    def pylon_amount(self, my_unit: Unit, dis=6.5):
        pylons = self.bot.structures(UnitTypeId.PYLON).filter(
            lambda unit: unit.distance_to_squared(my_unit) <= dis ** 2)
        return pylons.amount

    def warp_pylons(self) -> Units:
        warp_pylons: Units = Units([], self.bot)
        if self.bot.structures(UnitTypeId.WARPGATE).ready:
            warp_pylons = self.bot.structures(UnitTypeId.PYLON).filter(
                lambda unit: unit.distance_to(self.bot.structures(
                    UnitTypeId.WARPGATE).ready.closest_to(unit.position).position) <= 6.5).ready
        if self.bot.structures(UnitTypeId.NEXUS):
            warp_pylons.extend(self.bot.structures(UnitTypeId.PYLON).filter(
                lambda unit: unit.distance_to(self.bot.structures(
                    UnitTypeId.NEXUS).closest_to(unit.position).position) <= 7 and unit not in warp_pylons).ready)
        return warp_pylons



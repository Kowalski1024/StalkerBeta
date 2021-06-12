from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING

from CombatCommander.Roles.unit_task import UnitTask

from sc2 import BotAI
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.units import Units
from sc2.constants import *
from loguru import logger
import numpy as np
import pandas as pd
import random

if TYPE_CHECKING:
    from commander import Commander


class Scout:
    def __init__(self, bot: "Commander"):
        self.bot = bot
        self.scout_tag = None
        self.finished = True
        self.enemy_found = False
        self.enemy_base_points = 0
        self.target_move = None
        self.unseen_enemy_region = set()
        pass

    def scout_by_unit(self, sight_range: int = 1):
        scout_unit = self.bot.units.by_tag(self.scout_tag)
        if self.bot.iteration % 2 == 0:
            sight = int(scout_unit.sight_range) + sight_range
            pos = scout_unit.position
            pos_set = set()
            for x in range(-sight, sight):
                for y in range(-sight, sight):
                    if x ** 2 + y ** 2 <= sight ** 2:
                        p = tuple((round(pos.x) + x, round(pos.y) + y))
                        pos_set.add(p)
            self.unseen_enemy_region = self.unseen_enemy_region.difference(pos_set)

            if self.scout_tag not in self.bot.path_walker.unit_dic:
                self.target_move = self.unseen_enemy_region.pop()
                self.bot.path_walker.add_unit(scout_unit, action="move", target=self.target_move)

            if not self.unseen_enemy_region:
                self.scout_tag = None
                self.finished = True
                self.bot.scout_thread_alive = False
                self.unseen_enemy_region = self.bot.regions.enemy_base
                print("scout ended")
        pass

    def scout_by_worker(self):
        # choose worker
        if self.scout_tag is None:
            worker = self.bot.select_build_worker(self.bot.enemy_start_locations[0])
            self.scout_tag = worker.tag
            self.bot.path_walker.add_unit(worker, action="move", target=self.bot.enemy_start_locations[0])
            self.bot.unit_role.set_task(UnitTask.Moving, worker)
            print(self.bot.unit_role.task_dic)
        scout_worker: Unit = self.bot.workers.by_tag(self.scout_tag)
        # enemy base found!
        if (
                self.bot.enemy_structures
                and not self.enemy_found
        ):
            if self.scout_tag in self.bot.path_walker.unit_dic:
                self.bot.path_walker.unit_dic.pop(self.scout_tag)
            if self.enemy_base_points is None:
                self.enemy_found = True
                print("enemy found")
            elif (
                    len(self.bot.map_data.where_all(self.bot.enemy_start_locations[0].position)[0].buildables.points)
                    < self.enemy_base_points
            ):
                self.enemy_found = True
                self.finished = False
                print("enemy found")

        if scout_worker.distance_to(self.bot.enemy_start_locations[0]) < 2:
            self.bot.enemy_start_locations.pop(0)
        pass
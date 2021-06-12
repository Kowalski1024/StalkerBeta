from collections import deque
from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING
from CombatCommander.Roles.unit_task import UnitTask

from sc2.constants import *
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from MapAnalyzer import MapData

if TYPE_CHECKING:
    from MapAnalyzer import MapData
    from commander import Commander


class PathingWalker:
    def __init__(self, bot: "Commander", map_data: "MapData"):
        self.map_data = map_data
        self.bot = bot
        self.grid = None
        self.worker_grid = None
        self.unit_dic: Dict[int, list] = dict()

    async def update_path(self):
        self.grid = self.worker_grid = self.map_data.get_pyastar_grid()
        for enemy_unit in self.bot.enemy_units:
            self.worker_grid = self.map_data.add_cost(enemy_unit.position_tuple, enemy_unit.ground_range,
                                                      grid=self.worker_grid)

        for unit_tag in list(self.unit_dic):
            unit = self.bot.units.find_by_tag(unit_tag)
            if unit is None:
                continue
            target = self.unit_dic[unit_tag][1]
            if unit.type_id == UnitTypeId.PROBE:
                new_path = self.map_data.pathfind(unit.position, target, grid=self.worker_grid, sensitivity=4)
            else:
                new_path = self.map_data.pathfind(unit.position, target, grid=self.grid, sensitivity=2)
            if len(new_path) > 0:
                self.unit_dic[unit_tag][2] = new_path
            else:
                self.unit_dic.pop(unit_tag)
        pass

    async def move_all(self):
        for unit_tag in self.unit_dic:
            # if unit_tag in self.bot.unit_role.task_dic[UnitTask.Moving]:
            unit = self.bot.units.find_by_tag(unit_tag)
            action = self.unit_dic[unit_tag][0]
            path = self.unit_dic[unit_tag][2]
            target = path[0]
            dist = 1.5 * unit.calculate_speed() * 1.4
            if unit is None:
                continue
            if self.map_data.distance(unit.position, target) <= dist:
                path.pop(0)
                if len(path) > 0:
                    eval("unit."+action+"(Point2("+str(path[0])+"))")
        pass

    def add_unit(self, unit, action: str, target, start_pos: tuple = None):
        """
        :param unit:
        :param target:
        :param action: type of action: move, attack
        :param start_pos:
        :return:
        """
        if isinstance(target, Unit):
            target = target.position_tuple
        elif isinstance(target, Point2):
            target = (target.x, target.y)
        if start_pos is None:
            start_pos = unit.position
        # print(target)

        if unit.type_id == UnitTypeId.PROBE:
            self.unit_dic[unit.tag] = [action, target,
                                       self.map_data.pathfind(start_pos, target, grid=self.worker_grid)]
        else:
            self.unit_dic[unit.tag] = [action, target, self.map_data.pathfind(start_pos, target, grid=self.grid)]
        pass


class UnitPath:
    if TYPE_CHECKING:
        from commander import Commander

    def __init__(self, unit: Unit, action, target: Point2, bot: "Commander"):
        self.bot = bot
        self.target = target
        self.actual_target: Point2
        self.path = set
        self.action_type = action
        self.tag = unit.tag
        pass

    def get_path(self, grid, sens):
        unit = self.bot.units.find_by_tag(self.tag)
        self.path = self.bot.map_data.pathfind(unit.position, self.target, grid=grid, sensitivity=sens)

    def do_action(self, grid, sens):
        unit = self.bot.units.find_by_tag(self.tag)
        if unit is None:
            return

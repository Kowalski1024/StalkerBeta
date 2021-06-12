from typing import Union, Dict, Set, Deque, List, Optional, Iterable , TYPE_CHECKING

from sc2.position import Point2
from sc2.client import Client
from sc2.unit import Unit
from sc2.units import Units
from CombatCommander.Roles.unit_task import UnitTask
from CombatCommander.Roles.unit_in_role import UnitsInRole

if TYPE_CHECKING:
    from commander import Commander

# https://github.com/DrInfy/sharpy-sc2/blob/42c35aebe80d28f01f0ec816dcb93f67002c968e/sharpy/managers/core/unit_role_manager.py#L251


class UnitRoleManager:
    MAX_VALUE = 10

    def __init__(self, bot: "Commander"):
        self.bot = bot
        self.role_count = UnitRoleManager.MAX_VALUE
        self.task_dic: Dict[int, set] = dict()
        self.had_task_set: Set[int] = set()
        self.roles: List[UnitsInRole] = list()
        self.debug = True

    async def start(self):
        for index in range(0, self.role_count):
            self.roles.append(UnitsInRole(index, self.bot))

    async def update(self):
        pass

    def set_task(self, task: Union[int, UnitTask], unit: Unit):
        if unit.tag not in self.had_task_set:
            self.had_task_set.add(unit.tag)
        for i in range(0, self.role_count):
            if i == task:
                self.roles[i].register_unit(unit)
            else:
                self.roles[i].remove_unit(unit)

    def set_tasks(self, task: Union[int, UnitTask], units: Units):
        for unit in units:
            self.set_task(task, unit)

    def clear_task(self, unit: Union[Unit, int]):
        if isinstance(unit, int):
            unit = self.bot.units.find_by_tag(unit)
            if unit is None:
                return
        for i in range(0, self.role_count):
            if i == UnitTask.Idle:
                self.roles[i].register_unit(unit)
            else:
                self.roles[i].remove_unit(unit)

    def clear_tasks(self, units: Union[Units, Iterable[int]]):
        for unit in units:
            self.clear_task(unit)

    async def post_update(self):
        if self.debug:
            idle = len(self.roles[UnitTask.Idle].tags)
            building = len(self.roles[UnitTask.Building].tags)
            gathering = len(self.roles[UnitTask.Gathering].tags)
            scouting = len(self.roles[UnitTask.Scouting].tags)
            moving = len(self.roles[UnitTask.Moving].tags)
            fighting = len(self.roles[UnitTask.Fighting].tags)
            defending = len(self.roles[UnitTask.Defending].tags)
            attacking = len(self.roles[UnitTask.Attacking].tags)
            reserved = len(self.roles[UnitTask.Reserved].tags)
            hallucination = len(self.roles[UnitTask.Hallucination].tags)

            msg = (
                f"I{idle} B{building} G{gathering} S{scouting} M{moving} "
                f"F{fighting} D{defending} A{attacking} R{reserved} H{hallucination}"
            )

            for index in range(10, self.role_count):
                key = str(index)
                count = len(self.roles[index].tags)
                msg += f" {key}:{count}"

            client: Client = self.bot._client
            client.debug_text_screen(msg, Point2((0.01, 0.01)), None, 16)


from sc2.unit import Unit
from sc2.units import Units
from CombatCommander.Roles.unit_task import UnitTask
from typing import Dict, Set, Deque, List, Optional, Union, TYPE_CHECKING
from sc2 import BotAI

# https://github.com/DrInfy/sharpy-sc2/blob/42c35aebe80d28f01f0ec816dcb93f67002c968e/sharpy/managers/core/roles/units_in_role.py#L24


class UnitsInRole:
    def __init__(self, task: Union[int, UnitTask], bot: BotAI):
        self.bot = bot
        self.tags: Set[int] = set()
        self.task = task
        self.units: Units = Units([], bot)

    def clear(self):
        self.units.clear()
        self.tags.clear()

    def register_units(self, units: Units):
        for unit in units:
            self.register_unit(unit)

    def register_unit(self, unit: Unit):
        if unit.tag not in self.tags:
            self.units.append(unit)
            self.tags.add(unit.tag)

    def remove_units(self, units: Units):
        for unit in units:
            self.remove_unit(unit)

    def remove_unit(self, unit: Unit):
        if unit.tag in self.tags:
            self.units.remove(unit)
            self.tags.remove(unit.tag)

    def update(self):
        self.units.clear()
        new_tags: Set[int] = set()

        for tag in self.tags:
            unit = self.bot.units.find_by_tag(tag)
            if unit is not None:
                # update unit to collection
                self.units.append(unit)
                new_tags.add(tag)
        self.tags = new_tags


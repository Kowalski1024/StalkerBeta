from collections import deque
from typing import Dict, Set, Deque, List, Optional, TYPE_CHECKING

from sc2.constants import *
from sc2.unit import Unit
from sc2.units import Units
from .manager_base import ManagerBase

# from https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/managers/extensions/memory_manager.py

MAX_SNAPSHOTS_PER_UNIT = 10


class UnitMemory(ManagerBase):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        # units that bot saw recently
        self.memory_units_by_tag: Dict[int, Deque[Unit]] = dict()
        # units that the bot hasn't seen in a long time
        self.archive_units_by_tag: Dict[int, Unit] = dict()
        # all enemy units that are out of vision
        self.unit_dict: Dict[int, Unit] = dict()

        self.expire_air = 60  # Time in seconds when snapshot expires
        self.expire_ground = 60  # Time in seconds when snapshot expires

    async def update(self):
        units_left_vision = self.bot._enemy_units_previous_map

        for unit_tag in units_left_vision:
            if units_left_vision[unit_tag].type_id in ignored_unit_types:
                continue

            if unit_tag in self.archive_units_by_tag:
                self.archive_units_by_tag.pop(unit_tag)

            self.memory_units_by_tag[unit_tag] = units_left_vision[unit_tag]
            self.memory_units_by_tag[unit_tag] = self.memory_units_by_tag.get(unit_tag, deque(maxlen=3))
            self.unit_dict[unit_tag] = units_left_vision[unit_tag]

        memory_tags_to_remove = list()
        for unit_tag in self.memory_units_by_tag:
            snap = self.get_latest_snapshot(unit_tag)

            if self.check_expiration(snap):
                self.clear_unit_cache(memory_tags_to_remove, unit_tag)

        memory_units = self.ghost_units()
        self.bot.all_know_enemy_units.clear()
        self.bot.all_know_enemy_units = self.bot.enemy_units + memory_units

    async def post_update(self):
        pass

    # archive by last snapshot units that the bot hasn't seen in a long time
    def clear_unit_cache(self, memory_tags_to_remove, unit_tag):
        memory_tags_to_remove.append(unit_tag)
        snaps = self.memory_units_by_tag.get(unit_tag, deque(maxlen=1))
        self.archive_units_by_tag[unit_tag] = snaps

    def on_unit_destroyed(self, unit_tag):
        if unit_tag in self.memory_units_by_tag:
            self.memory_units_by_tag.pop(unit_tag)
        elif unit_tag in self.archive_units_by_tag:
            self.memory_units_by_tag.pop(unit_tag)
        elif unit_tag in self.unit_dict:
            self.unit_dict.pop(unit_tag)
        pass

    def get_latest_snapshot(self, unit_tag: int) -> Unit:
        unit = self.memory_units_by_tag.get(unit_tag, deque(maxlen=1))
        return unit

    def check_expiration(self, snap: Unit) -> bool:
        if snap.is_flying:
            return snap.age > self.expire_air
        return snap.age > self.expire_ground

    def is_unit_visible(self, unit_tag: int) -> bool:
        """Returns true if the unit is visible on this frame."""
        unit: Optional[Unit] = self.unit_dict.get(unit_tag)
        return unit is not None and not unit.is_memory

    def ghost_units(self) -> Units:
        """Returns latest snapshot for all units that we know of but which are currently not visible."""
        memory_units = Units([], self.bot)

        for tag in self.unit_dict:
            if self.is_unit_visible(tag):
                continue

            snap = self.get_latest_snapshot(tag)
            memory_units.append(snap)

        return memory_units


ignored_unit_types = {
    # Protoss
    UnitTypeId.INTERCEPTOR,
    # Terran
    UnitTypeId.MULE,
    UnitTypeId.AUTOTURRET,
    # Zerg
    UnitTypeId.LARVA,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.INFESTEDTERRAN,
    UnitTypeId.BROODLING,
}

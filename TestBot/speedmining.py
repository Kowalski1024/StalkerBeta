import sc2
from sc2.constants import *
from sc2 import Race
from sc2.player import Bot, Computer, Difficulty
import math
import numpy as np
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from typing import List, Union, Tuple
from Bot.Managers.townhalls_manager import Townhall
# from bot import BotBrain


class Speed(sc2.BotAI):
    townhall: Townhall

    def __init__(self):
        super().__init__()
        self.iteration = 0
        self.units_dict: Dict[int, Unit] = dict()
        self.workers_dict: Dict[int, Unit] = dict()

    async def on_unit_created(self, unit: Unit):
        if unit.type_id == UnitTypeId.PROBE:
            self.townhall.register_worker(unit)
        pass

    async def on_before_start(self):
        self.units_dict = {unit.tag: unit for unit in self.units}
        fields = self.mineral_field.filter(lambda unit: unit.distance_to(self.townhalls[0]) <= 10)
        self.townhall = Townhall(fields, self, self.townhalls[0])

    async def on_start(self):
        self.client.game_step = 8
        print("Game started")

    async def on_step(self, iteration):
        pass
        self.iteration = iteration
        print(self.step_time)
        for unit in self.units:
            self.units_dict[unit.tag] = unit
        await self.townhall.speed_mining()
        await self.townhall.mining_debug()
        if self.minerals > 6000:
            await self.client.quit()
        pass

    def on_end(self, result):
        print("Game ended.", result)


def main():
    # "DeathAura506"
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("DeathAura506"), [
        Bot(Race.Protoss, Speed()),
        # bot(Race.Protoss, BotBrain())
        Computer(Race.Protoss, Difficulty.VeryEasy)
    ], realtime=True, disable_fog=False, save_replay_as='mining.SC2Replay')


if __name__ == '__main__':
    main()

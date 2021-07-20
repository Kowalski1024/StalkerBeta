import sc2
from sc2.constants import *
from sc2 import Race
from sc2.player import Bot, Computer, Difficulty
from sc2.unit import Unit
from MapAnalyzer.MapData import MapData
import sc2_math

from Managers.build_order_manager import BuildOrderManager
from Managers.influence_gird_manager import InfluenceGridManager
from MapInfluence.grid_types import GridTypes


class TossBot(sc2.BotAI):
    def __init__(self):
        super().__init__()
        self.iteration = int
        self.map_data = MapData
        self.gridManager = InfluenceGridManager(self)
        self.buildManager = BuildOrderManager(self)

    async def on_unit_destroyed(self, unit_tag):
        pass

    async def on_building_construction_started(self, unit: Unit):
        await self.gridManager.on_building_construction_started(unit)
        pass

    async def on_building_construction_complete(self, unit: Unit):
        await self.gridManager.on_building_construction_complete(unit)
        pass

    async def on_before_start(self):
        # split the workers
        if self.townhalls.exists:
            mfs = self.mineral_field.closer_than(10, self.townhalls.random)
            for worker in self.units(UnitTypeId.PROBE):
                if len(mfs) > 0:
                    mf = mfs.closest_to(worker)
                    worker.gather(mf)
                    mfs.remove(mf)

    async def on_start(self):
        self.map_data = MapData(self)
        print(self.main_base_ramp.protoss_wall_pylon)
        self.gridManager.on_create(self.map_data)
        self.buildManager.on_create()
        print("Game started")

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers(resource_ratio=2)
        await self.buildManager.update()
        # if self.iteration % 10 == 0:
        #     if self.can_afford(UnitTypeId.PYLON):
        #         reg = (self.gridManager[GridTypes.Pylons] + self.gridManager[GridTypes.Placement] - self.gridManager[GridTypes.Power]) & self.gridManager[GridTypes.Buildings]
        #         target_pos = sc2_math.find_best_position(reg.grid, 2)
        #         print(target_pos)
        #         worker = self.select_build_worker(target_pos)
        #         worker.build(UnitTypeId.PYLON, target_pos)
        await self.buildManager.post_update()
        pass

    def on_end(self, result):
        print("Game ended.", result)


def main():
    # "AcropolisLE"
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("AcropolisLE"), [
        Bot(Race.Protoss, TossBot()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=True, disable_fog=False)


if __name__ == '__main__':
    main()

import sc2
from sc2.constants import *
from sc2 import Race
from sc2.player import Bot, Computer, Difficulty
from sc2.unit import Unit


class TossBot(Commander):
    def __init__(self):
        super().__init__()

    async def on_unit_destroyed(self, unit_tag):
        pass

    async def on_building_construction_started(self, unit: Unit):
        pass

    async def on_building_construction_complete(self, unit: Unit):
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
        await self.builder.train_probe()

    async def on_start(self):
        print("Game started")

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers(resource_ratio=2)
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

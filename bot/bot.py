import sc2
from sc2.constants import *
from sc2 import Race
from sc2.player import Bot, Computer, Difficulty
from sc2.unit import Unit
from MapAnalyzer.MapData import MapData


from Managers.townhalls_manager import Townhall

GAME_STEP = 8
DATA_PATH = "Data/"


class BotBrain(sc2.BotAI):
    map_data: MapData
    townhall: Townhall

    def __init__(self):
        super().__init__()
        self.iteration = 0
        self.units_dict: Dict[int, Unit] = dict()

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
        self.map_data = MapData(self)
        print("Game started")

    async def on_step(self, iteration):
        self.iteration = iteration
        for unit in self.units:
            self.units_dict[unit.tag] = unit
        await self.townhall.speed_mining()
        await self.townhall.mining_debug()
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
    # "AcropolisLE"
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("RomanticideAIE"), [
        Bot(Race.Protoss, BotBrain()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=True, disable_fog=False, )


if __name__ == '__main__':
    main()

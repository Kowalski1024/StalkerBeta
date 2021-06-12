import sc2
from sc2.constants import *
from sc2 import Race
from sc2.player import Bot, Computer, Difficulty
from sc2.unit import Unit
from MapAnalyzer import MapData
from archive.builder_brain import BuilderManager, Regions
from ProductionManager.builder import BuildManager
from ProductionManager.building_placer import Regions
from CombatCommander.scout import Scout
from commander import Commander

DATA_FILE_STALKER = 'DATA/stalker_data'


class TossBot(Commander):
    def __init__(self):
        super().__init__()
        self.builder = BuildManager(self)
        self.regions = Regions(self)
        self.scout = Scout(self)

    def on_unit_destroyed(self, unit_tag: int):
        if unit_tag == self.scout.scout_tag:
            self.scout.scout_tag = None
        # the less the more rewards
        mineral_weight = 600
        vespene_weight = 500

        dead_unit: Unit
        if unit_tag in self.prev_enemy_units_tags:
            dead_unit = self.prev_enemy_units.by_tag(unit_tag)
            self.prev_enemy_units_tags.remove(unit_tag)
            self.prev_enemy_units.remove(dead_unit)
            my_units_nearby = self.units.filter(
                lambda unit: dead_unit.distance_to_squared(unit.position) <= unit.sight_range**2)
            if my_units_nearby:
                cost_mineral = self.calculate_cost(dead_unit.type_id).minerals
                cost_vespene = self.calculate_cost(dead_unit.type_id).vespene
                self.reward_dict["unit_manager"] += round(cost_mineral/mineral_weight + cost_vespene/vespene_weight, 3)
        elif unit_tag in self.prev_units_tags:
            dead_unit = self.prev_units.by_tag(unit_tag)
            cost_mineral = self.calculate_cost(dead_unit.type_id).minerals
            cost_vespene = self.calculate_cost(dead_unit.type_id).vespene
            self.reward_dict["unit_manager"] -= round(cost_mineral / mineral_weight + cost_vespene / vespene_weight, 3)
        pass

    async def on_building_construction_started(self, unit: Unit):
        if unit.is_structure:
            self.regions.update_structures(unit)
            if unit.type_id == UnitTypeId.NEXUS:
                resources = self.resources.filter(lambda u: u.distance_to(unit) <= 10)
                self.regions.facing_region(self.regions.placement, resources, unit, dis=4,
                                           num=self.regions.reg_dic["resource_region"])
                await self.build_pylon()

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id == UnitTypeId.PYLON:
            self.regions.pylon_grid_update(unit)
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
        await self.train_probe()

    async def on_start(self):
        print("Game started")
        # placements girds
        self.regions = Regions(self)
        self.regions.update_structures()

        self.map_data = MapData(self)
        try:
            # creating lists of points enemy and bot main
            self.regions.main_base = self.map_data.where_all(self.townhalls[0].position)[0].buildables.points
            self.regions.enemy_base = self.map_data.where_all(self.enemy_start_locations[0].position)[
                0].buildables.points
            self.regions.enemy_unseen_region = self.scout.unseen_enemy_region = set(self.regions.enemy_base)
            self.scout.enemy_base_points = len(self.regions.enemy_base)

            # update main placement
            for x, y in self.regions.main_base:
                self.regions.placement[y, x] += 4
            print("Map data OK")
        except:
            self.special_units_dict["enemy_base_points"] = None
            print("Map data FAIL")

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers(resource_ratio=2)

        if self.ramp_exists:
            await self.build_ramp_wall()
        elif self.structures_dict["gateway"] == 0:
            self.structures_dict["cybernetics_core"] = True
            self.structures_dict["gateway"] = 1
            print("no ramp")

        # scout enemy base with worker
        if (
                not self.scout.enemy_found
                and self.structures(UnitTypeId.GATEWAY).exists
        ):
            self.scout.scout_by_worker()
        #
        # scout enemy base when scout_finished is False and has scout_unit
        if not self.scout.finished and self.scout.scout_tag is not None:
            self.scout.scout_by_unit()

        # builder manager
        await self.builder_manager()

        self.prev_units = self.units
        self.prev_enemy_units = self.enemy_units
        self.prev_units_tags = self.units.tags
        self.prev_enemy_units_tags = self.enemy_units.tags
        pass

    def on_end(self, result):
        print("Game ended.", result)
        # self.stalker_brain.stalker_QTable.learn(str(self.stalker_brain.previous_state),
        #                                         self.stalker_brain.previous_action,
        #                                         self.reward_dict["stalker"], 'terminal')
        # self.stalker_brain.stalker_QTable.q_table.to_pickle(DATA_FILE_STALKER + '.gz', 'gzip')
        # print("QLearningTable SAVED")


def main():
    # "AcropolisLE"
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    sc2.run_game(sc2.maps.get("AcropolisLE"), [
        Bot(Race.Protoss, TossBot()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=False, disable_fog=False)


if __name__ == '__main__':
    main()

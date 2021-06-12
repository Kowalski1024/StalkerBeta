import sc2
from collections import OrderedDict
from sc2 import BotAI, Race
from sc2.player import Bot, Computer, Human
from sc2.ids.unit_typeid import UnitTypeId
from sc2.game_state import GameState
from sc2.score import ScoreDetails
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.units import Units
from sc2.unit import Unit
from sc2.constants import *
from sc2.position import Point2, Point3
from sc2 import Race, Difficulty
from sc2.data import ActionResult, Attribute, Race
from sc2.player import Bot, Computer
import time
import random
import numpy as np
import pandas as pd
import os

from MyUnits.stalker import Stalker as My_stalker


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))


class TossBot(sc2.BotAI):

    actions = ("do_nothing",
               "train_adept",
               "train_stalker"
               )

    def __init__(self):
        sc2.BotAI.__init__(self)
        self.iteration = None
        self.ITERATIONS_PER_MINUTE = 165
        self.SAVE_MINERALS = 0
        self.MATCH_HAS_STARTED = False

        # QTable
        self.step = 0
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.qtable = QLearningTable(self.actions)
        self.previous_action = None
        self.previous_state = None

        self._unit_manager = My_stalker(self)

        # UNIT AMOUNT
        self.MAX_WORKERS = 50
        self.ASSIMILATOR_AMOUNT = 0

        # BUILDINGS
        self.PYLON_WALL_EXISTS = False
        self.GATEWAY_WALL_EXISTS = False
        self.RAMP_READY = False

        self.PYLON_WALL = None

        self.GATEWAYS_AMOUNT = 0
        self.STARGATES_AMOUNT = 0
        self.ROBOTICSFACILITY_AMOUNT = 0


        # TECH
        self.CYBERNETICSCORE_EXISTS = False
        self.CYBERNETICSCORE_READY = False

    def do_nothing(self):
        pass

    def get_unit_info(self, unit, field="build_time"):
        # get various unit data, see list below
        # usage: getUnitInfo(ROACH, "mineral_cost")
        assert isinstance(unit, (Unit, UnitTypeId))
        if isinstance(unit, Unit):
            unit = unit._type_data._proto
        else:
            unit = self._game_data.units[unit.value]._proto
        if hasattr(unit, field):
            return getattr(unit, field)

    async def check_buildings(self):

        # RAMP WALL
        cyberneticscore_wall_exists = False
        pylon_wall_position = self.main_base_ramp.protoss_wall_pylon
        if self.structures(UnitTypeId.PYLON).exists:
            pylon = self.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            if pylon.position == pylon_wall_position:
                self.PYLON_WALL_EXISTS = True
        buildings_wall_position: Set[Point2] = self.main_base_ramp.protoss_wall_buildings
        for d in buildings_wall_position:
            if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).exists:
                gateway = self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).closest_to(d)
                if gateway.position == d:
                    self.GATEWAY_WALL_EXISTS = True
            if self.structures(UnitTypeId.CYBERNETICSCORE).exists:
                cyberneticscore = self.structures(UnitTypeId.CYBERNETICSCORE).closest_to(d)
                if cyberneticscore.position == d:
                    cyberneticscore_wall_exists = True
        if self.GATEWAY_WALL_EXISTS and cyberneticscore_wall_exists and self.PYLON_WALL_EXISTS:
            self.RAMP_READY = True
        else:
            self.RAMP_READY = False

        # CYBERNETICSCORE
        self.CYBERNETICSCORE_READY = True if self.structures(UnitTypeId.CYBERNETICSCORE).ready else False
        self.CYBERNETICSCORE_EXISTS = True if self.structures(UnitTypeId.CYBERNETICSCORE).exists else False

    def build_fast_as_possible(self, unit_type, track_unit, target_building_location):
        workers: Units = self.units(UnitTypeId.PROBE)
        if workers:
            worker: Unit = workers.closest_to(target_building_location)
            travel_time = round(worker.distance_to(target_building_location) / worker.movement_speed, 2)
            building_time = self.getUnitInfo(track_unit.type_id, "build_time")
            building_end_time = round((building_time - building_time * track_unit.build_progress) / 22.4 + 1.5, 2)
            # await self.chat_send(str(travel_time) + " " + str(building_end_time))
            if (travel_time >= building_end_time
                    and track_unit.build_progress < 1
                    and self.iteration % 2 == 0
            ):
                worker.move(target_building_location)
            elif track_unit.build_progress == 1:
                self.do(worker.build(unit_type, target_building_location))

    async def build_ramp_wall(self):
        self.PYLON_WALL_EXISTS = False
        pylon_wall_position = self.main_base_ramp.protoss_wall_pylon
        if self.structures(UnitTypeId.PYLON).exists:
            pylon = self.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)
            if pylon.position == pylon_wall_position:
                self.PYLON_WALL_EXISTS = True
                self.PYLON_WALL = self.structures(UnitTypeId.PYLON).closest_to(pylon_wall_position)

        # First PYLON
        if (self.can_afford(UnitTypeId.PYLON)
                and self.already_pending(UnitTypeId.PYLON) == 0
                and not self.PYLON_WALL_EXISTS
        ):
            if len(pylon_wall_position) == 0:
                return
            workers: Units = self.units(UnitTypeId.PROBE)
            if workers:
                worker: Unit = workers.closest_to(pylon_wall_position)
                self.do(worker.build(UnitTypeId.PYLON, pylon_wall_position))

        buildings_wall_position: Set[Point2] = self.main_base_ramp.protoss_wall_buildings

        # First GATEWAY
        if (self.can_afford(UnitTypeId.GATEWAY)
                and self.PYLON_WALL_EXISTS
                and not self.GATEWAY_WALL_EXISTS
        ):
            if len(buildings_wall_position) == 0:
                return
            target_building_location: Point2 = buildings_wall_position.pop()
            self.build_fast_as_possible(UnitTypeId.GATEWAY, self.PYLON_WALL, target_building_location)

        # Clear GATEWAY position
        gateway_wall_position: Units = self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE})
        if gateway_wall_position:
            buildings_wall_position: Set[Point2] = {
                d for d in buildings_wall_position if gateway_wall_position.closest_distance_to(d) > 1
            }

        # Build CYBERNETICSCORE
        if (self.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).exists
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            gateway_wall = gateway_wall_position.random
            if len(buildings_wall_position) == 0:
                return
            target_building_location: Point2 = buildings_wall_position.pop()
            self.build_fast_as_possible(UnitTypeId.CYBERNETICSCORE, gateway_wall, target_building_location)

    async def train_probe(self):
        if ((len(self.structures(UnitTypeId.NEXUS)) * 22) > len(self.units(UnitTypeId.PROBE))
                and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS
        ):
            for nexus in self.structures(UnitTypeId.NEXUS).ready.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    self.do(nexus.train(UnitTypeId.PROBE))

    async def warp_new_units(self, proxy, unit_id):
        for warpgate in self.structures(UnitTypeId.WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            # all the units have the same cooldown anyway so let's just look at ZEALOT
            if AbilityId.WARPGATETRAIN_STALKER in abilities:
                pos = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    # return ActionResult.CantFindPlacementLocation
                    print("can't place")
                    return
                warpgate.warp_in(unit_id, placement)

    async def train_stalkers(self):
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if self.structures(UnitTypeId.WARPGATE).exists:
                for gateway in self.structures(UnitTypeId.GATEWAY).ready:
                    if (
                            self.can_afford(UnitTypeId.STALKER)
                            and gateway.is_idle
                            and self.get_unit_info(UnitTypeId.ZEALOT, "food_required") < self.supply_left
                    ):
                        gateway.train(UnitTypeId.STALKER)
            # elif self.structures(UnitTypeId.GATEWAY).exists:

                # self.warp_new_units(self, , UnitTypeId.STALKER)

    async def build_pylon(self):
        if self.supply_left < 3 and not self.already_pending(UnitTypeId.PYLON) and self.PYLON_WALL_EXISTS:
            nexuses = self.structures(UnitTypeId.NEXUS).ready
            pos = nexuses.first.position.towards_with_random_angle(self.enemy_start_locations[0], random.randint(4, 8))
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=pos)

    async def build_gateways(self):
        pylons = self.structures.filter(lambda structure: structure.type_id == UnitTypeId.PYLON
                                        and structure != self.PYLON_WALL
                                        and structure.is_ready
                                        )
        if (
                self.can_afford(UnitTypeId.GATEWAY)
                and pylons
                and self.GATEWAY_WALL_EXISTS
                and self.already_pending(UnitTypeId.GATEWAY) == 0
                and self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).amount < 3
        ):
            pylon = pylons.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15, nexus)
            vg = vgs.random
            if (self.structures(UnitTypeId.ASSIMILATOR).amount < self.ASSIMILATOR_AMOUNT
                    and not self.already_pending(UnitTypeId.ASSIMILATOR)
                    or self.structures(UnitTypeId.ASSIMILATOR).amount + 1 < self.ASSIMILATOR_AMOUNT
            ):
                worker = self.select_build_worker(vg.position)
                if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                    self.do(worker.build(UnitTypeId.ASSIMILATOR, vg))
                    worker.stop(queue=True)

    async def on_start(self):
        print("Game started")
        # Do things here before the game starts

    async def on_match_start(self):

        # split the workers on start
        self.MATCH_HAS_STARTED = True
        if self.townhalls.exists:
            mfs = self.mineral_field.closer_than(10, self.townhalls.random)
            for worker in self.units(UnitTypeId.PROBE):
                if len(mfs) > 0:
                    mf = mfs.closest_to(worker)
                    worker.gather(mf)
                    mfs.remove(mf)

    async def on_step(self, iteration):
        self.iteration = iteration

        if not self.MATCH_HAS_STARTED:
            await self.on_match_start()

        if iteration % 10 == 0:
            await self.check_buildings()
        await self.distribute_workers()
        await self.build_assimilators()
        await self.build_pylon()
        await self.build_gateways()

        if self.SAVE_MINERALS + 50 <= self.minerals:
            await self.train_probe()

        # build ramp
        if not self.RAMP_READY:
            await self.build_ramp_wall()
        # build one gas
        if self.structures(UnitTypeId.GATEWAY).exists:
            self.ASSIMILATOR_AMOUNT = 1
        # research WARPGATE
        if (self.CYBERNETICSCORE_READY
                and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0
                and self.can_afford(AbilityId.RESEARCH_WARPGATE)
        ):
            ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            ccore.research(UpgradeId.WARPGATERESEARCH)
        # expand natural
        if (self.CYBERNETICSCORE_EXISTS
                and self.townhalls.ready.amount + self.already_pending(UnitTypeId.NEXUS) < 2
        ):
            if self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
        pass

    def on_end(self, result):
        print("Game ended.")
        # Do things here after the game ends


def main():
    sc2.run_game(sc2.maps.get("AcropolisLE"), [
        Bot(Race.Protoss, TossBot()),
        Computer(Race.Protoss, Difficulty.VeryEasy)
    ], realtime=False)


if __name__ == '__main__':
    main()

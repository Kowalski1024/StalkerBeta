import sc2
from sc2.units import Units
from sc2.unit import Unit
from sc2.constants import *
from sc2.position import Point2
from sc2 import Difficulty
from sc2.data import Race
from sc2.player import Bot, Computer
from archive import stalker_brain
import random
import numpy as np
import pandas as pd

DATA_FILE = 'refined_agent_data'

STALKER_AMOUNT = 2

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


class myBot(sc2.BotAI):
    def __init__(self):
        sc2.BotAI.__init__(self)
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50

        # self.unit_command_uses_self_do = True
        self.enemy_base_location = 0
        self.scout_unit = None

    async def do_nothing(self):
        pass

    async def train_probe(self):
        if (len(self.structures(UnitTypeId.NEXUS)) * 22) > len(self.units(UnitTypeId.PROBE)) \
                and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.structures(UnitTypeId.NEXUS).ready.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)

    async def build_pylon(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.structures(UnitTypeId.NEXUS).ready
            pos = nexuses.first.position.towards(self.enemy_start_locations[0], random.randint(5, 10))
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=pos)

    async def build_gateways(self):
        if (
                self.can_afford(UnitTypeId.GATEWAY)
                and self.structures(UnitTypeId.PYLON).ready
                and self.already_pending(UnitTypeId.GATEWAY) == 0
                and self.structures(UnitTypeId.GATEWAY).amount < 1
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_cyberneticscore(self):
        if (
                self.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.structures(UnitTypeId.GATEWAY).exists
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def build_assimilators(self):
        if self.structures(UnitTypeId.GATEWAY).exists:
            for nexus in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(15, nexus)
                for vg in vgs:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vg.position)
                    if worker is None:
                        break;
                    if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                        worker.build(UnitTypeId.ASSIMILATOR, vg)
                        worker.stop(queue=True)

    async def train_zealots(self):
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            for gateway in self.structures(UnitTypeId.GATEWAY).ready:
                if (
                        self.can_afford(UnitTypeId.STALKER)
                        and gateway.is_idle
                        and self.units(UnitTypeId.STALKER).amount <= STALKER_AMOUNT
                ):
                    gateway.train(UnitTypeId.STALKER)

    async def on_start(self):
        pass

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.train_probe()
        await self.build_pylon()
        await self.build_cyberneticscore()
        await self.train_zealots()
        await self.build_gateways()
        await self.build_assimilators()
        if self.units(UnitTypeId.STALKER).amount > STALKER_AMOUNT and self.enemy_units(UnitTypeId.STALKER).amount > STALKER_AMOUNT:
            for stalker in self.units(UnitTypeId.STALKER):
                stalker.attack(Point2((60, 60)))

        pass

    def on_end(self, result):
        pass


class TossBot(stalker_brain.StalkersBrain):
    actions = ("do_nothing",
               "back",
               "attack_low_hp",
               "attack_close"
               )

    def __init__(self):
        stalker_brain.StalkersBrain.__init__(self)
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50
        self.READY = False
        # self.unit_command_uses_self_do = True
        self.myStalkers = set()
        self.enemy_tags = set()
        self.stalker_target = set()

        self.step = 0
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.qtable = QLearningTable(self.actions)
        self.previous_action = None
        self.previous_state = None

        self.enemy_base_location = 0
        self.scout_unit = None
        # if os.path.isfile(DATA_FILE + '.gz'):
        #     self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
        #     print("QLearningTable LOADED")

    #

    async def do_nothing(self, obs, unit):
        pass

    async def train_probe(self):
        if (len(self.structures(UnitTypeId.NEXUS)) * 22) > len(self.units(UnitTypeId.PROBE)) \
                and len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.structures(UnitTypeId.NEXUS).ready.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)

    async def build_pylon(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.structures(UnitTypeId.NEXUS).ready
            pos = nexuses.first.position.towards(self.enemy_start_locations[0], random.randint(5, 10))
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                   await self.build(UnitTypeId.PYLON, near=pos)

    async def build_gateways(self):
        if (
                self.can_afford(UnitTypeId.GATEWAY)
                and self.structures(UnitTypeId.PYLON).ready
                and self.already_pending(UnitTypeId.GATEWAY) == 0
                and self.structures(UnitTypeId.GATEWAY).amount < 1
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_cyberneticscore(self):
        if (
                self.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.structures(UnitTypeId.GATEWAY).exists
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
                and self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1
        ):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def build_assimilators(self):
        if self.structures(UnitTypeId.GATEWAY).exists:
            for nexus in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(15, nexus)
                for vg in vgs:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vg.position)
                    if worker is None:
                        break
                    if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                        worker.build(UnitTypeId.ASSIMILATOR, vg)
                        worker.stop(queue=True)

    async def train_zealots(self):
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            for gateway in self.structures(UnitTypeId.GATEWAY).ready:
                if (
                        self.can_afford(UnitTypeId.STALKER)
                        and gateway.is_idle
                        and self.units(UnitTypeId.STALKER).amount <= STALKER_AMOUNT
                ):
                    gateway.train(UnitTypeId.STALKER)

    async def attack(self, obs):
        enemies: Units = self.enemy_units | self.enemy_structures
        enemies_can_attack: Units = enemies.filter(lambda unit: unit.can_attack_ground)
        for stalker in self.units(UnitTypeId.STALKER):
            enemyThreatsClose: Units = enemies_can_attack.filter(
                lambda unit: unit.target_in_range(stalker)
            )

            if stalker.health_percentage < 4 / 5 and enemyThreatsClose:
                retreatPoints: Set[Point2] = self.neighbors8(stalker.position, distance=2) | self.neighbors8(
                    stalker.position, distance=4
                )

                retreatPoints: Set[Point2] = {x for x in retreatPoints if self.in_pathing_grid(x)}
                if retreatPoints:
                    closestEnemy: Unit = enemyThreatsClose.closest_to(stalker)
                    retreatPoint: Unit = closestEnemy.position.furthest(retreatPoints)
                    stalker.move(retreatPoint)
                    continue

            enemyGroundUnits: Units = enemies.filter(
                lambda unit: unit.distance_to(stalker) <= stalker.target_in_range(unit) and not unit.is_flying
            )  # Hardcoded attackrange of 5
            if stalker.weapon_cooldown == 0 and enemyGroundUnits:
                enemyGroundUnits: Units = enemyGroundUnits.sorted(lambda x: x.distance_to(stalker))
                closestEnemy: Unit = enemyGroundUnits[0]
                stalker.attack(closestEnemy)
                continue

            enemyThreatsVeryClose: Units = enemies.filter(
                lambda unit: unit.can_attack_ground
                and unit.target_in_range(stalker, -1)
            )  # Hardcoded attackrange minus 0.5
            # Threats that can attack the reaper
            # stalker.weapon_cooldown != 0 and
            if enemyThreatsVeryClose:
                retreatPoints: Set[Point2] = self.neighbors8(stalker.position, distance=2) | self.neighbors8(
                    stalker.position, distance=4
                )
                # Filter points that are pathable by a reaper
                retreatPoints: Set[Point2] = {x for x in retreatPoints if self.in_pathing_grid(x)}
                if retreatPoints:
                    closestEnemy: Unit = enemyThreatsVeryClose.closest_to(stalker)
                    retreatPoint: Point2 = max(
                        retreatPoints, key=lambda x: x.distance_to(closestEnemy) - x.distance_to(stalker)
                    )
                    stalker.move(retreatPoint)
                    continue  # Continue for loop, don't execute any of the following

            # Move to nearest enemy ground unit/building because no enemy unit is closer than 5
            allEnemyGroundUnits: Units = self.enemy_units.not_flying
            if allEnemyGroundUnits:
                closestEnemy: Unit = allEnemyGroundUnits.closest_to(stalker)
                stalker.move(closestEnemy)
                continue  # Continue for loop, don't execute any of the following

            # Move to random enemy start location if no enemy buildings have been seen
            stalker.move(random.choice(self.enemy_start_locations))
        # stalkers = self.units(UnitTypeId.STALKER).ready
        # for stalker in stalkers:
        #     stalker.attack(self.enemy_start_locations[0])

    # Stolen and modified from position.py
    def neighbors4(self, position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return {Point2((p.x - d, p.y)), Point2((p.x + d, p.y)), Point2((p.x, p.y - d)), Point2((p.x, p.y + d))}

    def neighbors8(self, position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }

    async def on_unit_destroyed(self, unit_tag: int):
        if unit_tag in self.enemy_tags:
            print("enemy dead")
            self.enemy_tags.remove(unit_tag)

        # if unit_tag in self.myStalkers:
        #     print("dead")
        #     self.myStalkers.remove(unit_tag)
        pass

    async def on_start(self):
        print("Game started")

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.train_probe()
        await self.build_pylon()
        await self.build_cyberneticscore()
        await self.train_zealots()
        await self.build_gateways()
        await self.build_assimilators()
        self.draw_visibility_pixelmap()
        for enemy in self.enemy_units:
            if enemy.tag not in self.enemy_tags:
                self.enemy_tags.add(enemy.tag)

        for stalker in self.units(UnitTypeId.STALKER):
            if stalker.tag not in self.myStalkers:
                self.myStalkers.add(stalker.tag)
        await self.choose_action_stalker(iteration)
        # if self.units(UnitTypeId.STALKER).amount > STALKER_AMOUNT:

        # # await self.attack()
        # for stalker in self.units(UnitTypeId.STALKER):
        #     self.stalker_target.add(stalker.order_target)
        #     if iteration % 10 == 1:
        #         print(self.stalker_target)
        #     # stalker.attack(Point2((60, 60)))
        #     state = str(self.get_state(iteration))
        #     action = self.qtable.choose_action(state)
        #
        #     if self.previous_action is not None:
        #         self.qtable.learn(self.previous_state,
        #                           self.previous_action,
        #                           0,
        #                           state)
        #
        #     self.previous_state = state
        #     self.previous_action = action
        #     return await getattr(self, action)(iteration, stalker)

        pass

    def on_end(self, result):
        print("Game ended.", result)
        self.step = 2
        reward = -1
        # self.previous_action = None
        # self.previous_state = None
        # Do things here after the game ends

    def get_state(self, obs):
        time = self.time
        summary = self.state.score
        probes = self.units(UnitTypeId.PROBE)
        idle_probes = [probe for probe in probes if probe.is_idle]
        nexuses = self.townhalls.ready
        pylons = self.structures(UnitTypeId.PYLON)
        completed_pylons = self.structures(
            UnitTypeId.PYLON).ready
        gateways = self.structures(UnitTypeId.GATEWAY)
        completed_gateways = self.structures(UnitTypeId.GATEWAY).ready
        zealots = self.units(UnitTypeId.STALKER)

        return (time,
                summary,
                len(nexuses),
                len(probes),
                len(idle_probes),
                len(pylons),
                len(completed_pylons),
                len(gateways),
                len(completed_gateways),
                len(zealots))
        # free_supply,
        # can_afford_supply_depot,
        # can_afford_barracks,
        # can_afford_marine,
        # len(enemy_command_centers),
        # len(enemy_scvs),
        # len(enemy_idle_scvs),
        # len(enemy_supply_depots),
        # len(enemy_completed_supply_depots),
        # len(enemy_barrackses),
        # len(enemy_completed_barrackses),
        # len(enemy_marines))

def main():
    sc2.run_game(sc2.maps.get("Flat96"), [
        Bot(Race.Protoss, TossBot()),
        # Bot(Race.Protoss, myBot()),
        Computer(Race.Protoss, Difficulty.VeryEasy),
    ], realtime=False, disable_fog=False)


if __name__ == '__main__':
    main()

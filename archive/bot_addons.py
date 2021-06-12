import sc2
from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.unit import Unit
from sc2.constants import *
from sc2.game_data import Cost
from sc2 import Race, Difficulty
from sc2.data import ActionResult, Attribute, Race
from loguru import logger
from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
import random
from typing import List
import numpy as np
import pandas as pd
import os
from threading import Thread

GREEN = Point3((0, 255, 0))
RED = Point3((255, 0, 0))
BLUE = Point3((0, 0, 255))
BLACK = Point3((0, 0, 0))


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


class BotState(sc2.BotAI):
    def __init__(self):
        super().__init__()

        # Game data
        self.iteration = 0
        self.ramp_exists = True
        self.prev_enemy_units: Units = Units([], self)
        self.prev_units: Units = Units([], self)
        self.prev_enemy_units_tags = set()
        self.prev_units_tags = set()
        # self.structures_under_attack = Units([], self)

        self.unit_manager = None
        self.scout = None
        self.regions = None

        self.base_region = None

        # Map Analyzer
        self.map_data = None
        self.enemy_base_region = None
        self.enemy_base_unseen_region = None

        # Bot data
        self.save_minerals = 0
        self.supply_need = 3

        self.special_units_dict = {
            "scout_tag": None,
            "scout_finished": True,
            "enemy_found": False,
            "enemy_base_points": 0,
            "target_point": tuple()
        }

        self.reward_dict = {
            "unit_manager": float(0),
        }

        self.structures_dict = {
            "nexuses": 0,
            "pylons": 0,
            "assimilators": 0,
            "gateway": 0,
            "stargate": 0,
            "robotics_facility": 0,
            "forge": 0,
            "cybernetics_core": False,
            "robotics_bay": False,
            "twilight_council": False,
            "templar_archives": False,
            "dark_shrine": False,
            "fleet_beacon": False
        }

    async def on_step(self, iteration):
        pass

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

    def list_of_structures(self, enemy_structures: Units, enemy: bool = True):
        neutral_structures_list = list()
        defense_structures_list = list()
        if enemy:
            race = self.enemy_race
        else:
            race = Race.Protoss
        if race == Race.Random:
            return None
        if race == Race.Protoss:
            defense_structures_types = [
                {UnitTypeId.SHIELDBATTERY},
                {UnitTypeId.PHOTONCANNON}
            ]
            neutral_structures_types = [
                {UnitTypeId.SHIELDBATTERY},
                {UnitTypeId.PHOTONCANNON},
                {UnitTypeId.NEXUS},
                {UnitTypeId.PYLON},
                {UnitTypeId.ASSIMILATOR},
                {UnitTypeId.GATEWAY, UnitTypeId.WARPGATE},
                {UnitTypeId.FORGE},
                {UnitTypeId.CYBERNETICSCORE},
                {UnitTypeId.TWILIGHTCOUNCIL},
                {UnitTypeId.TEMPLARARCHIVE},
                {UnitTypeId.DARKSHRINE},
                {UnitTypeId.ROBOTICSFACILITY},
                {UnitTypeId.ROBOTICSBAY},
                {UnitTypeId.STARGATE},
                {UnitTypeId.FLEETBEACON}
            ]
            for structure_type in defense_structures_types:
                defense_structures_list.append(len(enemy_structures.of_type(structure_type).filter(
                    lambda unit: unit.is_powered)))
            for structure_type in neutral_structures_types:
                neutral_structures_list.append(len(enemy_structures.of_type(structure_type)))
            return defense_structures_list, neutral_structures_list
        return None
        pass

    def list_of_units(self, enemy_units: Units, enemy: bool = True):
        unit_list = list()
        if enemy:
            race = self.enemy_race
        else:
            race = Race.Protoss
        if race == Race.Random:
            return None
        if race == Race.Protoss:
            units_types = [
                {UnitTypeId.PROBE},
                {UnitTypeId.ZEALOT},
                {UnitTypeId.SENTRY},
                {UnitTypeId.STALKER},
                {UnitTypeId.ADEPT},
                {UnitTypeId.HIGHTEMPLAR},
                {UnitTypeId.DARKTEMPLAR},
                {UnitTypeId.ARCHON},
                {UnitTypeId.OBSERVER},
                {UnitTypeId.WARPPRISM, UnitTypeId.WARPPRISMPHASING},
                {UnitTypeId.IMMORTAL},
                {UnitTypeId.COLOSSUS},
                {UnitTypeId.DISRUPTOR, UnitTypeId.DISRUPTORPHASED},
                {UnitTypeId.PHOENIX},
                {UnitTypeId.ORACLE},
                {UnitTypeId.VOIDRAY},
                {UnitTypeId.TEMPEST},
                {UnitTypeId.CARRIER},
                {UnitTypeId.MOTHERSHIP}
            ]
            for unit_type in units_types:
                unit_list.append(len(enemy_units.of_type(unit_type)))
            return unit_list
        return None

    # basic information about my unit
    def get_state_for_unit(self, my_unit: Unit):
        unit_state = list()
        enemy_near = self.all_enemy_units.filter(
            lambda unit: unit.distance_to_squared(my_unit.position) <= my_unit.sight_range**2)
        friends_near = self.all_own_units.filter(
            lambda unit: unit.distance_to_squared(my_unit.position) <= my_unit.sight_range**2)

        # shield, hp, weapon_ready
        unit_state.extend([round(my_unit.health_percentage*100), round(my_unit.shield_percentage*100),
                           int(my_unit.weapon_ready)])

        # TODO: my upgrades and the enemy

        # hp weakest enemy at range
        weak_enemy = enemy_near.filter(
            lambda unit: my_unit.target_in_range(unit)).sorted(lambda unit: unit.shield_health_percentage)
        if weak_enemy:
            unit_state.append(round(weak_enemy.first.shield_health_percentage*100))
        else:
            unit_state.append(-1)

        # distance to nearest enemy
        close_enemy = enemy_near.filter(
            lambda unit: my_unit.target_in_range(unit))
        if close_enemy:
            unit_state.append(round(close_enemy.closest_to(my_unit).distance_to(my_unit.position)))
        else:
            unit_state.append(-1)

        # friendly units near
        friendly_units = self.list_of_units(friends_near, enemy=False)
        friendly_units.extend(self.list_of_structures(friends_near, enemy=False)[0])
        unit_state.extend(friendly_units)

        # enemies units near
        enemy_units_at_sights_range = self.list_of_units(enemy_near)
        enemy_units_at_sights_range.extend(self.list_of_structures(enemy_near)[0])
        unit_state.extend(enemy_units_at_sights_range)

        # enemies that my unit is in their range
        # TODO: add terran structures with bonus range
        enemy_units_at_their_attack_range = self.list_of_units(enemy_near.filter(
            lambda unit: unit.target_in_range(my_unit)))
        enemy_units_at_their_attack_range.extend(self.list_of_structures(enemy_near.filter(
            lambda unit: unit.target_in_range(my_unit)))[0])
        unit_state.extend(enemy_units_at_their_attack_range)

        # dps my units and enemies
        # nearest enemies at range with bonus dmg vs them
        # sum of enemies dps at range vs my unit
        # distance to my main and enemy

        return unit_state
        pass

    # def get_state_for_building(self):
    #     pass

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

    def draw_lines_unit_to_units(self, from_unit: Unit, to_units: Units, color: Point3):
        p0 = from_unit.position3d
        if not to_units:
            return
        to_unit: Unit
        for to_unit in to_units:
            if to_unit == from_unit:
                continue
            p1 = to_unit.position3d
            # Red
            self._client.debug_line_out(p0, p1, color=color)
        pass

    def draw_point_list(self, point_list: List = None, color=None, text=None, box_r=None):
        if not color:
            color = GREEN
        h = self.get_terrain_z_height(self.townhalls[0])
        for p in point_list:
            p = Point2(p)

            pos = Point3((p.x, p.y, h))
            if box_r:
                p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r))# + Point2((0.5, 0.5))
                p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r))# + Point2((0.5, 0.5))
                self.client.debug_box_out(p0, p1, color=color)
            if text:
                self.client.debug_text_world(
                    "\n".join([f"{text}", ]), pos, color=color, size=30,
                )


class MyBotAddons(BotState):
    def __init__(self):
        super().__init__()

    async def on_step(self, iteration):
        pass

    # gives pylons with nexus and ready warpgate
    def warp_pylons(self) -> Units:
        _warp_pylons: Units = Units([], self)
        if self.structures(UnitTypeId.WARPGATE).ready:
            _warp_pylons = self.structures(UnitTypeId.PYLON).filter(
                lambda unit: unit.distance_to(self.structures(
                    UnitTypeId.WARPGATE).ready.closest_to(unit.position).position) <= 6.5).ready
        if self.structures(UnitTypeId.NEXUS):
            _warp_pylons.extend(self.structures(UnitTypeId.PYLON).filter(
                lambda unit: unit.distance_to(self.structures(
                    UnitTypeId.NEXUS).closest_to(unit.position).position) <= 7 and unit not in _warp_pylons).ready)

        return _warp_pylons

    # overwrite function from BotAI
    def neighbors4(self, position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return {Point2((p.x - d, p.y)), Point2((p.x + d, p.y)), Point2((p.x, p.y - d)), Point2((p.x, p.y + d))}

    # overwrite function from BotAI
    def neighbors8(self, position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }

    # overwrite function from BotAI
    async def distribute_workers(self, resource_ratio: float = 2):
        """
        Distributes workers across all the bases taken.
        Keyword `resource_ratio` takes a float. If the current minerals to gas
        ratio is bigger than `resource_ratio`, this function prefer filling gas_buildings
        first, if it is lower, it will prefer sending workers to minerals first.

        NOTE: This function is far from optimal, if you really want to have
        refined worker control, you should write your own distribution function.
        For example long distance mining control and moving workers if a base was killed
        are not being handled.

        WARNING: This is quite slow when there are lots of workers or multiple bases.

        :param resource_ratio: """
        if not self.mineral_field or not self.workers or not self.townhalls.ready:
            return
        worker_pool = [worker for worker in self.workers.idle]
        bases = self.townhalls.ready
        gas_buildings = self.gas_buildings.ready

        # list of places that need more workers
        deficit_mining_places = []

        for mining_place in bases | gas_buildings:
            difference = mining_place.surplus_harvesters
            # perfect amount of workers, skip mining place
            if not difference:
                continue
            if mining_place.has_vespene:
                # get all workers that target the gas extraction site
                # or are on their way back from it
                local_workers = self.workers.filter(
                    lambda unit: unit.order_target == mining_place.tag
                                 or (unit.is_carrying_vespene and unit.order_target == bases.closest_to(
                        mining_place).tag)
                )

            else:
                # get tags of minerals around expansion
                local_minerals_tags = {
                    mineral.tag for mineral in self.mineral_field if mineral.distance_to(mining_place.position) <= 8
                }
                # get all target tags a worker can have
                # tags of the minerals he could mine at that base
                # get workers that work at that gather site
                local_workers = self.workers.filter(
                    lambda unit: unit.order_target in local_minerals_tags
                                 or (unit.is_carrying_minerals and unit.order_target == mining_place.tag)
                )

            if len(gas_buildings) == 1 and gas_buildings[0].assigned_harvesters < 3 and not self.vespene:
                for worker in local_workers:
                    worker_pool.append(worker)

            # too many workers
            if difference > 0:
                for worker in local_workers[:difference]:
                    worker_pool.append(worker)
            # too few workers
            # add mining place to deficit bases for every missing worker
            else:
                deficit_mining_places += [mining_place for _ in range(-difference)]

        # prepare all minerals near a base if we have too many workers
        # and need to send them to the closest patch
        if len(worker_pool) > len(deficit_mining_places):
            all_minerals_near_base = [
                mineral
                for mineral in self.mineral_field
                if any(mineral.distance_to(base.position) <= 8 for base in self.townhalls.ready)
            ]
        # distribute every worker in the pool
        for worker in worker_pool:
            # as long as have workers and mining places
            if deficit_mining_places:
                # choose only mineral fields first if current mineral to gas ratio is less than target ratio
                if self.vespene != 0 or self.vespene and self.minerals / self.vespene < resource_ratio:
                    possible_mining_places = [place for place in deficit_mining_places if
                                              not place.vespene_contents]
                # else prefer gas
                else:
                    possible_mining_places = [place for place in deficit_mining_places if place.vespene_contents]
                # if preferred type is not available any more, get all other places
                if not possible_mining_places:
                    possible_mining_places = deficit_mining_places
                # find closest mining place
                current_place = min(deficit_mining_places, key=lambda place: place.distance_to(worker.position))
                # remove it from the list
                deficit_mining_places.remove(current_place)
                # if current place is a gas extraction site, go there
                if current_place.vespene_contents:
                    worker.gather(current_place)
                # if current place is a gas extraction site,
                # go to the mineral field that is near and has the most minerals left
                else:
                    local_minerals = (
                        mineral for mineral in self.mineral_field if mineral.distance_to(current_place) <= 8
                    )
                    # local_minerals can be empty if townhall is misplaced
                    target_mineral = max(local_minerals, key=lambda mineral: mineral.mineral_contents, default=None)
                    if target_mineral:
                        worker.gather(target_mineral)
            # more workers to distribute than free mining spots
            # send to closest if worker is doing nothing
            elif worker.is_idle and all_minerals_near_base:
                target_mineral = min(all_minerals_near_base, key=lambda mineral: mineral.distance_to(worker.position))
                worker.gather(target_mineral)
            else:
                # there are no deficit mining places and worker is not idle
                # so dont move him
                pass

    # functions from BotAI
    async def train_unit(
            self, unit_type: UnitTypeId, amount: int = 1, closest_to: Point2 = None,
            train_only_idle_buildings: bool = True
    ) -> int:
        """ Trains a specified number of units. Trains only one if amount is not specified.
        Warning: currently has issues with warp gate warp ins


        Example distance to::

            # If you want to train based on distance to a certain point, you can use "closest_to"
            self.train(UnitTypeId.MARINE, 4, closest_to = self.game_info.map_center)


        :param unit_type:
        :param amount:
        :param closest_to:
        :param train_only_idle_buildings: """
        # Tech requirement not met
        if self.tech_requirement_progress(unit_type) < 1:
            race_dict = {
                Race.Protoss: PROTOSS_TECH_REQUIREMENT,
            }
            unit_info_id = race_dict[self.race][unit_type]
            logger.warning(
                "{} Trying to produce unit {} in self.train() but tech requirement is not met: {}".format(
                    self.time_formatted, unit_type, unit_info_id
                )
            )
            return 0

        # Not affordable
        if not self.can_afford(unit_type):
            return 0

        trained_amount = 0
        # All train structure types: queen can made from hatchery, lair, hive
        train_structure_type: Set[UnitTypeId] = UNIT_TRAINED_FROM[unit_type]
        train_structures = self.structures
        is_protoss = self.race == Race.Protoss
        # Sort structures closest to a point
        if closest_to is not None:
            train_structures = train_structures.sorted_by_distance_to(closest_to.position)

        structure: Unit
        for structure in train_structures:
            # Exit early if we can't afford
            if not self.can_afford(unit_type):
                return trained_amount
            if (
                    # If structure hasn't received an action/order this frame
                    structure.tag not in self.unit_tags_received_action
                    # If structure can train this unit at all
                    and structure.type_id in train_structure_type
                    # Structure has to be completed to be able to train
                    and structure.build_progress == 1
                    # If structure is protoss, it needs to be powered to train
                    and (not is_protoss or structure.is_powered)
                    # Either parameter "train_only_idle_buildings" is False or
                    # structure is idle or structure has less than 2 orders and has reactor
                    and (
                    not train_only_idle_buildings
                    or len(structure.orders) < 1 + int(structure.add_on_tag in self.reactor_tags)
                    )
            ):
                successfully_trained = False
                # Warp in at location
                if structure.type_id == UnitTypeId.WARPGATE:
                    abilities = await self.get_available_abilities([structure])
                    if AbilityId.WARPGATETRAIN_STALKER in abilities:
                        pylons = self.warp_pylons()
                        if not pylons:
                            pylons = self.structures(UnitTypeId.PYLON)
                        if closest_to is not None:
                            pylons = pylons.sorted_by_distance_to(closest_to.position)
                        trained_bool = False
                        while pylons and not trained_bool:
                            location = pylons[0].position.to2.random_on_distance(5)
                            placement = await self.find_placement(UnitTypeId.PYLON,
                                                                  location, placement_step=2)
                            if placement is None:
                                pylons.pop(0)
                            else:
                                trained_bool = successfully_trained = structure.warp_in(unit_type, location)
                else:
                    # Normal train a unit from larva or inside a structure
                    successfully_trained = self.do(
                        structure.train(unit_type), subtract_cost=True, subtract_supply=True, ignore_warning=True
                    )
                    # Check if structure has reactor: queue same unit again
                    if (
                            # Check if we have enough cost or supply for this unit type
                            self.can_afford(unit_type)
                            # Structure needs to be idle in the current frame
                            and not structure.orders
                            # We are at least 2 away from goal
                            and trained_amount + 1 < amount
                    ):
                        trained_amount += 1
                        # With one command queue=False and one queue=True,
                        # you can queue 2 marines in a reactored barracks in one frame
                        successfully_trained = self.do(
                            structure.train(unit_type, queue=True),
                            subtract_cost=True,
                            subtract_supply=True,
                            ignore_warning=True,
                        )

                if successfully_trained:
                    trained_amount += 1
                    if trained_amount == amount:
                        # Target unit train amount reached
                        return trained_amount
                else:
                    # Some error occured and we couldn't train the unit
                    print("error")
                    return trained_amount
        return trained_amount

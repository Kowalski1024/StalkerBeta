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
from sc2.position import Point2, Point3, Rect
from sc2 import Race, Difficulty
from sc2.data import ActionResult, Attribute, Race
from sc2.player import Bot, Computer
import time
import random
import numpy as np
import pandas as pd
import os
from bot_addons import MyBotAddons, QLearningTable

DATA_FILE = 'DATA/stalker_data'


class StalkersBrain:
    actions = ("do_nothing",
               "back",
               "attack_low_hp",
               "attack_nearest_unit",
               "attack_air_unit",
               "attack_enemy_workers"
               )

    def __init__(self):
        self.enemies_units_nearby = None
        self.enemies_units_in_range = None
        self.enemy_workers = None
        self.friends_units_nearby = None

        # QLearning things
        self.reward = 0
        self.stalker_QTable = QLearningTable(self.actions, learning_rate=0.01, reward_decay=0.9)
        self.previous_action = None
        self.previous_state = None
        if os.path.isfile(DATA_FILE + '.gz'):
            self.stalker_QTable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
            print("Stalker data LOADED")

    # STALKER #
    # TODO attack unit with bonus dmg vs him
    # TODO attack and back off when weapon on cooldown
    # TODO blink

    async def do_nothing(self, bot, my_unit: Unit):
        pass

    async def stalker_manager(self, bot: MyBotAddons, stalker_unit: Unit):
        self.enemies_units_nearby = bot.enemy_units.filter(
            lambda unit: unit.distance_to(stalker_unit.position) <= stalker_unit.sight_range)
        self.enemy_workers = self.enemies_units_nearby.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE})
        self.enemies_units_nearby.exclude_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE})
        self.enemies_units_in_range = self.enemies_units_nearby.filter(
            lambda unit: stalker_unit.target_in_range(unit)
        )
        self.friends_units_nearby = bot.units.filter(
            lambda unit: unit.distance_to(stalker_unit.position) <= stalker_unit.sight_range)

        if self.enemies_units_in_range:
            self.reward = bot.reward_dict["stalker"]
            state = str(self.get_state(bot, stalker_unit))
            action = self.stalker_QTable.choose_action(state)

            if self.previous_action is not None:
                self.stalker_QTable.learn(self.previous_state, self.previous_action, self.reward, state)
            if self.reward:
                print(self.reward)
            bot.reward_dict["stalker"] = self.reward = 0
            self.previous_state = state
            self.previous_action = action
            return await getattr(self, action)(bot=bot, my_unit=stalker_unit)
        else:
            return False

    # back from enemy
    async def back(self, bot, my_unit: Unit):
        enemies: Units = self.enemies_units_nearby.filter(lambda unit: unit.target_in_range(my_unit, 2))
        if enemies:
            retreat_points: Set[Point2] = bot.neighbors8(my_unit.position, distance=2) | bot.neighbors8(
                my_unit.position, distance=4
            )
            retreat_points: Set[Point2] = {x for x in retreat_points if bot.in_pathing_grid(x)}
            if retreat_points:
                closest_enemy: Unit = enemies.closest_to(my_unit)
                retreat_points: Unit = closest_enemy.position.furthest(retreat_points)
                my_unit.move(retreat_points)
        pass

    async def attack_enemy_workers(self, bot, my_unit: Unit):
        if self.enemy_workers:
            my_unit.attack(self.enemy_workers.closest_to(my_unit))
        pass

    # attack low hp unit
    async def attack_low_hp(self, bot, my_unit: Unit):
        if self.enemies_units_in_range:
            weak_enemy = self.enemies_units_in_range.sorted(lambda unit: unit.shield_health_percentage)
            if weak_enemy:
                my_unit.attack(weak_enemy.first)
        pass

    async def attack_nearest_unit(self, bot, my_unit: Unit):
        nearest_units = self.enemies_units_nearby
        nearest_units.extend(self.enemy_workers)
        if nearest_units:
            enemies_can_attack: Units = nearest_units.filter(
                lambda unit: unit.can_attack_ground)
            if enemies_can_attack:
                closed_enemy = enemies_can_attack.closest_to(my_unit.position)
                if closed_enemy:
                    my_unit.attack(closed_enemy)
        pass

    async def attack_air_unit(self, bot, my_unit: Unit):
        flying_enemies: Units = self.enemies_units_in_range.filter(lambda unit: unit.is_flying)
        if not flying_enemies:
            return
        # 1: attack observer and raven
        if flying_enemies.of_type({UnitTypeId.OBSERVER, UnitTypeId.RAVEN}):
            my_unit.attack(flying_enemies.of_type({UnitTypeId.OBSERVER, UnitTypeId.RAVEN}).first)
        # 2: attack enemies who can attack ground
        elif flying_enemies.filter(lambda unit: unit.can_attack_ground):
            my_unit.attack(flying_enemies.sorted(lambda unit: unit.ground_dps).first)
        # 3: attack enemies who can attack air
        elif flying_enemies.filter(lambda unit: unit.can_attack_air):
            my_unit.attack(flying_enemies.sorted(lambda unit: unit.air_dps).first)
        # 4: attack enemies has cargo
        elif flying_enemies.filter(lambda unit: unit.cargo_max > 0):
            my_unit.attack(flying_enemies.sorted(lambda unit: unit.cargo_used).first)
        # 5: attack others
        else:
            my_unit.attack(flying_enemies.first)

    def get_state(self, bot, my_unit: Unit):
        return bot.get_state_for_unit(my_unit)
        pass

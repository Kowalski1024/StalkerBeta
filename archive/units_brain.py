from archive.bot_addons import MyBotAddons, QLearningTable
from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.constants import *

DATA_FILE_UNIT_MANAGER = 'DATA/unit_manager_data'


class UnitsManager:
    def __init__(self):
        pass

    async def units_manager(self, bot: MyBotAddons):
        pass


class UnitManager:
    actions = {
        "do_nothing"
    }

    def __init__(self, bot: MyBotAddons):
        self.bot = bot
        self.enemies_units_nearby: Units = Units([], bot)
        self.enemies_units_in_range: Units = Units([], bot)
        self.enemies_bonus_dmg: Units = Units([], bot)
        self.enemy_workers: Units = Units([], bot)

        # QLearning
        self.reward = 0
        self.previous_action = None
        self.previous_state = None
        self.unit_QTable = QLearningTable(self.actions, learning_rate=0.01, reward_decay=0.9)
        # if os.path.isfile(DATA_FILE_UNIT_MANAGER + '.gz'):
        #     self.unit_QTable.q_table = pd.read_pickle(DATA_FILE_UNIT_MANAGER + '.gz', compression='gzip')
        #     print("Unit data LOADED")
        pass

    async def unit_manager(self, my_unit: Unit):
        self.enemies_units_nearby = self.bot.enemy_units.filter(
            lambda unit: unit.distance_to(my_unit.position) <= my_unit.sight_range
        ).sorted(lambda unit: unit.shield_health_percentage)
        self.enemies_units_in_range = self.enemies_units_nearby.filter(
            lambda unit: my_unit.target_in_range(unit)
        )
        if self.enemies_units_in_range:
            self.enemies_bonus_dmg = CompactUnitState(my_unit).bonus_dmg_filter(self.enemies_units_in_range)
            self.enemy_workers = self.enemies_units_in_range.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE})
            self.enemies_units_in_range.exclude_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE})
            self.reward = self.bot.reward_dict["unit_manager"]
            state = str(self.get_state(my_unit))
            action = self.unit_QTable.choose_action(state)

            if self.previous_action is not None:
                self.unit_QTable.learn(self.previous_state, self.previous_action, self.reward, state)
            if self.reward:
                print(self.reward)
            self.bot.reward_dict["unit_manager"] = self.reward = 0
            self.previous_state = state
            self.previous_action = action
            return await getattr(self, action)(my_unit=my_unit)
        else:
            return False
        pass

    async def do_nothing(self, my_unit: Unit):
        pass

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

    async def attack_enemy_workers(self, my_unit: Unit):
        if self.enemy_workers:
            my_unit.attack(self.enemy_workers.first)
        pass

    async def attack_low_hp(self, my_unit: Unit):
        if self.enemies_units_in_range:
            my_unit.attack(self.enemies_units_in_range.first)
        pass

    async def attack_nearest_unit(self, my_unit: Unit):
        nearest_units = self.enemies_units_in_range
        nearest_units.extend(self.enemy_workers)
        if nearest_units:
            enemies_can_attack: Units = nearest_units.filter(
                lambda unit: not my_unit.is_flying and unit.can_attack_ground
                or my_unit.is_flying and unit.can_attack_air)
            if enemies_can_attack:
                closed_enemy = enemies_can_attack.closest_to(my_unit.position)
                if closed_enemy:
                    my_unit.attack(closed_enemy)
            else:
                closed_enemy = nearest_units.closest_to(my_unit.position)
                if closed_enemy:
                    my_unit.attack(closed_enemy)
        pass

    async def attack_air_unit(self, my_unit: Unit):
        if my_unit.can_attack_air:
            flying_enemies: Units = self.enemies_units_in_range.filter(lambda unit: unit.is_flying)
            if not flying_enemies:
                return
            # 1: attack observer and raven
            if flying_enemies.of_type({UnitTypeId.OBSERVER, UnitTypeId.RAVEN}):
                my_unit.attack(flying_enemies.of_type({UnitTypeId.OBSERVER, UnitTypeId.RAVEN}).first)
                return
            # 2: attack enemies who can attack my unit
            enemies_aa = flying_enemies.filter(lambda unit: unit.can_attack_air)
            enemies_ag = flying_enemies.filter(lambda unit: unit.can_attack_ground)
            if my_unit.is_flying and enemies_aa:
                bonus_dmg = CompactUnitState(my_unit).bonus_dmg_filter(enemies_aa)
                if bonus_dmg:
                    my_unit.attack(bonus_dmg.first)
                else:
                    my_unit.attack(enemies_aa.first)
                return
            elif not my_unit.is_flying and enemies_ag:
                bonus_dmg = CompactUnitState(my_unit).bonus_dmg_filter(enemies_ag)
                if bonus_dmg:
                    my_unit.attack(bonus_dmg.first)
                else:
                    my_unit.attack(enemies_ag.first)
                return
            # 3: attack armed units
            flying_enemies_can_attack = flying_enemies.filter(
                lambda unit: unit.can_attack_air or unit.can_attack_ground
            )
            if flying_enemies_can_attack:
                my_unit.attack(flying_enemies_can_attack.first)
                return
            # 4: attack others
            my_unit.attack(flying_enemies.first)

    async def attack_unit_with_cargo(self, my_unit: Unit):
        enemy_with_cargo = self.enemies_units_in_range.filter(lambda unit: unit.cargo_max > 0)
        if enemy_with_cargo:
            my_unit.attack(enemy_with_cargo.sorted(
                lambda unit: CompactUnitState(unit).cargo_state(), reverse=True).first
            )

    async def attack_unit_with_bonus_dmg(self, my_unit: Unit):
        if self.enemies_bonus_dmg:
            my_unit.attack(self.enemies_bonus_dmg.first)

    def get_state(self, my_unit: Unit) -> str:
        state_list = list()

        # basic information about my unit
        state_list.extend([
            # hp and shield (0-3)
            CompactUnitState(my_unit).health_state(),
            CompactUnitState(my_unit).shield_state(),
            # weapon ready (bool)
            int(my_unit.weapon_ready),
            # is flying (bool)
            int(my_unit.is_flying),
            # can attack ground (bool)
            int(my_unit.can_attack_ground),
            # can attack air (bool)
            int(my_unit.can_attack_air),
            # Attribute (bool)
            int(my_unit.is_light),
            int(my_unit.is_biological),
            int(my_unit.is_massive),
            # is revealed (bool)
            int(my_unit.is_revealed)
        ])
        # nearest unit (distance)
        # enemy has flying or ground units
        if self.enemies_units_in_range.filter(lambda unit: unit.is_flying):
            state_list.append(1)
        else:
            state_list.append(0)
        if self.enemies_units_in_range.filter(lambda unit: not unit.is_flying):
            state_list.append(1)
        else:
            state_list.append(0)
        # enemy unit with health < 25% (bool)
        weak_enemy = self.enemies_units_in_range.filter(lambda unit: unit.shield_health_percentage <= 0.25)
        if weak_enemy:
            state_list.append(1)
        else:
            state_list.append(0)
        # high value unit (?)

        # unit with cargo (bool)
        units_with_cargo = self.enemies_units_in_range.filter(
            lambda unit: unit.cargo_max > 0 and round(unit.cargo_used/unit.cargo_max*100) >= 25
        )
        if units_with_cargo:
            state_list.append(1)
        else:
            state_list.append(0)
        # unit with bonus dmg (bool)
        if self.enemies_bonus_dmg:
            state_list.append(1)
        else:
            state_list.append(0)
        # any workers (bool)
        if self.enemy_workers:
            state_list.append(1)
        else:
            state_list.append(0)
        return str(state_list)
        pass


class Scout:
    def __init__(self, bot: MyBotAddons):
        self.bot = bot
        self.scout_tag = None
        self.finished = True
        self.enemy_found = False
        self.enemy_base_points = 0
        self.target_move = None
        self.unseen_enemy_region = set()
        pass

    def scout_by_unit(self, sight_range: int = 1):
        scout_unit = self.bot.units.by_tag(self.scout_tag)
        if self.bot.iteration % 2 == 0:
            sight = int(scout_unit.sight_range) + sight_range
            pos = scout_unit.position
            pos_set = set()
            for x in range(-sight, sight):
                for y in range(-sight, sight):
                    if x ** 2 + y ** 2 <= sight ** 2:
                        p = tuple((round(pos.x) + x, round(pos.y) + y))
                        pos_set.add(p)
            self.unseen_enemy_region = self.unseen_enemy_region.difference(pos_set)

            if self.target_move is None:
                self.target_move = self.unseen_enemy_region.pop()
            else:
                scout_unit.move(Point2(self.target_move))

            if self.unseen_enemy_region:
                if scout_unit.distance_to_squared(self.target_move) <= 4**2:
                    self.target_move = self.unseen_enemy_region.pop()
            else:
                self.scout_tag = None
                self.finished = True
                self.bot.scout_thread_alive = False
                self.unseen_enemy_region = self.bot.regions.enemy_base
                print("scout ended")
        pass

    def scout_by_worker(self):
        # choose worker
        if self.scout_tag is None:
            self.scout_tag = self.bot.select_build_worker(self.bot.enemy_start_locations[0]).tag
        scout_worker: Unit = self.bot.workers.by_tag(self.scout_tag)
        if self.bot.iteration % 10 == 0:
            scout_worker.move(self.bot.enemy_start_locations[0])

        # enemy base found!
        if (
                self.bot.enemy_structures
                and not self.enemy_found
        ):
            if self.enemy_base_points is None:
                self.enemy_found = True
                print("enemy found")
            elif (
                    len(self.bot.map_data.where_all(self.bot.enemy_start_locations[0].position)[0].buildables.points)
                    < self.enemy_base_points
            ):
                self.enemy_found = True
                self.finished = False
                print("enemy found")

        if scout_worker.distance_to(self.bot.enemy_start_locations[0]) < 2:
            self.bot.enemy_start_locations.pop(0)
        pass


class CompactUnitState:
    def __init__(self, my_unit: Unit):
        self.my_unit = my_unit

    def health_state(self) -> int:
        health = round(self.my_unit.health_percentage*100)
        if health <= 25:
            return 0
        elif health <= 50:
            return 1
        elif health <= 75:
            return 2
        else:
            return 3

    def shield_state(self) -> int:
        health = round(self.my_unit.shield_percentage*100)
        if health <= 5:
            return 0
        elif health <= 25:
            return 1
        elif health <= 50:
            return 2
        else:
            return 3

    def bonus_dmg_filter(self, enemy_units: Units) -> Units:
        bonus_damage = self.my_unit.bonus_damage[1]
        enemy_bonus_dmg = enemy_units.filter(
            lambda unit: unit.is_light and bonus_damage == 'Light'
            or unit.is_armored and bonus_damage == 'Armored'
            or unit.is_biological and bonus_damage == 'Biological'
            or unit.is_mechanical and bonus_damage == 'Mechanical'
            or unit.is_psionic and bonus_damage == 'Psionic'
            or unit.is_massive and bonus_damage == 'Massive'
        )
        return enemy_bonus_dmg

    def cargo_state(self) -> int:
        cargo_percentage = round(self.my_unit.cargo_used/self.my_unit.cargo_max*100)
        if cargo_percentage <= 33:
            return 1
        elif cargo_percentage <= 66:
            return 2
        else:
            return 3


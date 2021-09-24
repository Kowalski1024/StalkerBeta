from typing import TYPE_CHECKING, List, Union, Dict, Set, Optional
from dataclasses import dataclass

from sc2.game_info import GameInfo
from sc2.bot_ai import BotAI
from sc2.client import Client
from sc2.units import Units
from sc2.unit import Unit
from sc2.position import Point2, Point3
from sc2.cache import property_immutable_cache, property_mutable_cache
import sc2math
import matplotlib.pyplot as plt
from skimage.draw import line
from collections import deque
import numpy as np
import math
import random

from MapRegions.Region import Region
from MapRegions.constructs import Ramp, Blocker
from MapRegions.connectivity import ConnectivitySide


class MapRegions:
    def __init__(self, bot: BotAI):
        self.cache = {}
        self.bot = bot
        # game information
        self._game_info = self.bot.game_info
        self._client = self.bot.client
        self.destructables = self.bot.destructables
        self.mineral_blockers = self.bot.mineral_field.filter(
            lambda m: any(x in m.name.lower() for x in {"rich", "450"}))
        self.watchtowers: Units = self.bot.watchtowers
        self.expansion_locations = list(self.bot.expansion_locations_dict.keys())
        self.bot_start_loc = self.bot.start_location
        self.enemy_start_loc = self.bot.enemy_start_locations[0]

        # numpy grids
        self.terrain_height = self._game_info.terrain_height.data_numpy.copy()
        self.placement_grid = self._game_info.placement_grid.data_numpy.copy()
        self.pathing_grid = self._game_info.pathing_grid.data_numpy.copy()
        self.map_shape = self.pathing_grid.shape

        self.ramps = self._find_ramps()
        self.regions: Dict[int, Region] = self.slice_map()
        self.regions_map = np.zeros(self.map_shape)
        for reg in self.regions.values():
            self.regions_map[reg.array] = reg.label
        self._connect_regions()
        self.plot_regions()

    def slice_map(self):
        regions = []
        map_center = self._game_info.map_center
        region_grid = self.region_grid(self.placement_grid.copy(), 2, True)

        # bases locations
        for loc in self.expansion_locations:
            if not (self.bot_start_loc.is_same_as(loc, 1) or self.enemy_start_loc.is_same_as(loc, 1)):
                self.add_chokes(self.find_chokes(loc, distance=24), region_grid)
            reg = Region(self.bfs_region(loc.rounded, region_grid), self.expansion_locations, self.watchtowers)
            regions.append(reg)
        ramps_points = []
        for ramp in self.ramps:
            ramps_points.append(ramp.upper_side_point())
            ramps_points.append(ramp.lower_side_point())

        # sort ramps by distance to reflection point/line
        basses_center = Point2.center([self.bot_start_loc, self.enemy_start_loc])
        if basses_center.is_same_as(map_center, 1):
            ramps_points = map_center.sort_by_distance(ramps_points)
        else:
            ramps_points = sorted(ramps_points, key=lambda p: np.linalg.norm(
                np.cross(basses_center - map_center, map_center - p)) / np.linalg.norm(
                basses_center - map_center))

        for point in ramps_points:
            if sc2math.point2_in_grid(point, region_grid):
                self.add_chokes(self.find_chokes(point), region_grid)
                reg = Region(self.bfs_region(point.rounded, region_grid), self.expansion_locations, self.watchtowers)
                regions.append(reg)

        region_grid[region_grid == 2] = 1
        for (b, a), value in np.ndenumerate(region_grid):
            p = Point2((a, b))
            if value > 0 and sc2math.point2_in_grid(p, region_grid):
                regions.append(Region(self.bfs_region(p.rounded, region_grid),
                                      self.expansion_locations, self.watchtowers))
        regions = sorted(self._clean_regions(regions), key=lambda x: self.bot_start_loc.distance_to_point2(x.center))
        for idx, reg in enumerate(regions):
            reg.label = idx + 1
        return {reg.label: reg for reg in regions}

    def _clean_regions(self, regions, smallest_area: int = 100) -> List[Region]:
        def neighbours(_reg: Region, _grid: np.ndarray):
            neighbours_set = set()
            p = list(_reg.points)[0]
            val = _grid[p.y][p.x]
            deq = deque([p])
            region_points = np.zeros(_grid.shape, dtype=bool)
            while deq:
                p: Point2 = deq.popleft()
                if not region_points[p.y][p.x]:
                    region_points[p.y][p.x] = True
                    if _grid[p.y][p.x] == val:
                        deq.extend(p.neighbors8)
                    elif _grid[p.y][p.x] > 0:
                        neighbours_set.add(_grid[p.y][p.x])
            return neighbours_set

        map_size = self._game_info.placement_grid.data_numpy.shape
        regions_grid = np.zeros(map_size)
        regions_dict: Dict[int, Region] = dict()
        for idx, reg in enumerate(regions):
            regions_dict[idx + 1] = reg
            regions_grid[reg.array] = idx + 1

        regs = []
        regions = sorted(regions, key=lambda p: p.area)
        for reg in regions:
            if reg.area < smallest_area:
                neighbour_regs: List[Region] = [
                    regions_dict[v] for v in neighbours(reg, regions_grid)
                ]
                non_base_regs = [
                    neighbour for neighbour in neighbour_regs
                    if not neighbour.has_base and neighbour.area >= smallest_area
                ]
                if non_base_regs:
                    neighbour_regs = non_base_regs
                elif reg.area >= 50:
                    regs.append(reg)
                    continue
                above = [neighbour for neighbour in neighbour_regs if neighbour.area >= smallest_area]
                under = [neighbour for neighbour in neighbour_regs if neighbour.area < smallest_area]

                if above:
                    smallest_reg = min(above, key=lambda x: x.area)
                else:
                    if not under:
                        regs.append(reg)
                        continue
                    smallest_reg = min(under, key=lambda x: x.area)
                smallest_reg._update(reg)
            else:
                regs.append(reg)
        return regs

    def _connect_regions(self):
        # creating a perimeter grid of all regions
        regions_perimeter_grid = np.zeros(self.map_shape, dtype=np.uint8)
        for val, reg in self.regions.items():
            regions_perimeter_grid[reg.perimeter_as_indices] = val
        regions_perimeter_grid = regions_perimeter_grid.T

        for val, reg in self.regions.items():
            d: Dict[int, set] = {v: set() for v in self.regions.keys()}
            to_del = set()
            for p in reg.perimeter_as_points2:
                for n in p.neighbors4:
                    v = regions_perimeter_grid[n]
                    if v and v != val:
                        d[v].update({n, p})
                        to_del.add(n)
                        regions_perimeter_grid[p] = 0
            for p in to_del:
                regions_perimeter_grid[p] = 0
            for v, s in d.items():
                if s:
                    groups = self._game_info._find_groups(s, 4)
                    for group in groups:
                        neighbour = self.regions[v]
                        g1 = {p for p in group if reg.is_inside_point(p)}
                        g2 = {p for p in group if neighbour.is_inside_point(p)}
                        self.add_connectivity(reg, neighbour, g1, g2, False, False)

        for ramp in self.ramps:
            upper_regions: Set[Region] = set()
            lower_regions: Set[Region] = set()
            for p in ramp.upper:
                v = regions_perimeter_grid[p]
                if v:
                    upper_regions.add(self.regions[v])
                    regions_perimeter_grid[p] = 0
            for p in ramp.lower:
                v = regions_perimeter_grid[p]
                if v:
                    lower_regions.add(self.regions[v])
                    regions_perimeter_grid[p] = 0
            for reg_up in upper_regions:
                for reg_lw in lower_regions:
                    self.add_connectivity(reg_up, reg_lw, ramp.upper, ramp.lower, False, False)

    def add_connectivity(self, reg_a: Region, reg_b: Region,
                         points_a: Set[Point2], points_b: Set[Point2],
                         impassable: bool, jumpable: bool,
                         blocker: Optional[Blocker] = None):
        def add(r: Region, c: ConnectivitySide):
            label = c.neighbour.label
            if label not in r.connectivity_dict:
                r.connectivity_dict[label] = [c]
            else:
                r.connectivity_dict[label].append(c)

        connect_a = ConnectivitySide(self, reg_a, None, points_a, impassable, jumpable, blocker)
        connect_b = ConnectivitySide(self, reg_b, connect_a, points_b, impassable, jumpable, blocker)
        connect_a.neighbour_side = connect_b

        add(reg_a, connect_a)
        add(reg_b, connect_b)

    def region_grid(self, grid: np.ndarray, val: int, rich=False) -> np.ndarray:
        from MapAnalyzer.utils import change_destructable_status_in_grid

        # destructables
        constructs_grid = np.zeros(self.map_shape, dtype=bool).T
        for unit in self.destructables:
            change_destructable_status_in_grid(constructs_grid, unit, status=True)
        grid[constructs_grid.T] = False

        # mineral blockers
        resource_blockers = [m.position for m in self.mineral_blockers if "450" in m.name.lower()]
        if rich:
            resource_blockers.extend([m.position for m in self.mineral_blockers if "rich" in m.name.lower()])
        for pos in resource_blockers:
            x = int(pos.x) - 1
            y = int(pos.y)
            if grid[y, x]:
                grid[y, x:(x + 2)] = False

        # vision blockers
        constructs_grid = np.zeros(self.map_shape, dtype=bool)
        vision_blockers = sc2math.points2_to_indices(self._game_info.vision_blockers)
        constructs_grid[vision_blockers] = True
        constructs_indices = sc2math.points2_to_indices(sc2math.ndarray_corners_points2(constructs_grid))
        grid[constructs_indices] = val

        return grid

    @property_mutable_cache
    def choke_grid(self) -> np.ndarray:
        grid = self.pathing_grid.copy()
        grid[np.nonzero(self.placement_grid)] = True
        return self.region_grid(grid, 0)

    def _find_ramps(self):
        def height_levels(_p):
            return sorted(list({self.get_terrain_z_height(x) for x in _p}))

        from MapAnalyzer.utils import change_destructable_status_in_grid

        heights = {self.get_numpy_terrain_height(p) for p in self.expansion_locations}
        _min, _max = min(heights), max(heights)
        map_area = self._game_info.playable_area

        grid = self.pathing_grid.copy()

        # destructables
        constructs_grid = np.zeros(self.map_shape, dtype=bool).T
        for unit in self.destructables | self.mineral_blockers:
            change_destructable_status_in_grid(constructs_grid, unit, status=True)
        grid[constructs_grid.T] = True

        points = {
            Point2((a, b))
            for (b, a), value in np.ndenumerate(grid)
            if (
                    value == 1
                    and map_area.x <= a < map_area.x + map_area.width
                    and map_area.y <= b < map_area.y + map_area.height
                    and _min < self.terrain_height[b, a] < _max
                    and self.terrain_height[b, a] not in heights
            )
        }
        grid[np.nonzero(self.placement_grid)] = False
        points_groups = [group for group in self._game_info._find_groups(points) if len(height_levels(group)) >= 4]
        return [Ramp(self.bfs_array_to_points2(list(group)[0], grid), self._game_info)
                for group in points_groups
                ]

    def depth_points(self, center: Point2, dis, degrees_step) -> List[Point2]:
        grid = self.choke_grid.T
        vec = center.direction_vector(self._game_info.map_center).normalized
        vec_arr = np.array([[vec.x], [vec.y]])
        point_list = []
        for deg in range(0, 360, degrees_step):
            x, y = np.matmul(sc2math.rotation_matrix(deg), vec_arr).round(5)
            x, y = x[0], y[0]
            for m in range(1, dis):
                x_pos, y_pos = int(y * m + center.x), int(x * m + center.y)
                if not grid[x_pos, y_pos]:
                    point_list.append((x_pos, y_pos))
                    break

        # clean list from duplicates
        point_list = list(dict.fromkeys(point_list))
        # clean list from outliers
        distance = [math.hypot(p[0] - center.x, p[1] - center.y) for p in point_list]
        data = sc2math.get_outliers(np.array(distance))[0]
        data = {i for i in data if distance[i] > np.mean(np.array(distance))}
        return [Point2(i) for j, i in enumerate(point_list) if j not in data]

    def find_chokes(self, center: Union[tuple, Unit, Point2], distance: int = 25, degrees_step: int = 3):
        if isinstance(center, Unit):
            center = center.position
        elif isinstance(center, tuple):
            center = Point2(center)

        ps = self.depth_points(center, distance, degrees_step)
        graph: Dict[Point2, List[Point2]] = dict()

        for idx, p in enumerate(ps):
            graph[p] = [(ps[(idx + 1) % len(ps)])]

        for idx, p in enumerate(ps):
            prev_p = ps[idx - 1]
            next_p = ps[(idx + 1) % len(ps)]
            center_dist = p.distance_to_point2(center)
            prev_dist = p.distance_to_point2(prev_p)
            next_dist = p.distance_to_point2(next_p)
            ll = math.radians(degrees_step) * center_dist * 2
            if prev_dist >= ll + 1:
                for i in range(2, int(len(ps) / 4)):
                    dis = p.distance_to_point2(ps[idx - i])
                    if dis <= prev_dist:
                        prev_dist = dis
                        prev_p = ps[idx - i]
                graph[prev_p].append(p)

            if next_dist >= ll + 1:
                for i in range(2, int(len(ps) / 4)):
                    dis = p.distance_to_point2(ps[(idx + i) % len(ps)])
                    if dis <= next_dist:
                        next_dist = dis
                        next_p = ps[(idx + i) % len(ps)]
                graph[p].append(next_p)

        for idx, p in enumerate(ps):
            next_p = ps[(idx + 1) % len(ps)]
            if next_p in graph[p][1:]:
                for i in range(1, 8):
                    pos = ps[(idx + i + 1) % len(ps)]
                    if pos in graph[ps[(idx + i) % len(ps)]][1:]:
                        next_p = pos
                    else:
                        break
                graph[p].append(next_p)

        closest_point = min(ps, key=lambda x: x.distance_to_point2(center))
        p = closest_point
        chokes = []
        while True:
            if len(graph[p]) > 1:
                next_p = min(graph[p], key=lambda x: x.distance_to_point2(center))
                chokes.append((p, next_p))
                p = next_p
            else:
                p = graph[p][0]
            if p == closest_point:
                break

        # DEBUG
        # self.plot_depth(ps, graph, chokes, center)
        return chokes

    @staticmethod
    def add_chokes(chokes, grid: np.ndarray):
        for choke in chokes:
            x1, y1 = choke[0][1], choke[0][0]
            x2, y2 = choke[1][1], choke[1][0]
            rr, cc = line(x1, y1, x2, y2)
            for x, y in zip(rr, cc):
                if grid[x, y]:
                    grid[x, y] = 2

    # @staticmethod
    # def find_group(p: Point2, ps: Set[Point2]) -> Set[Point2]:
    #     s = set(p)
    #     visited = set()
    #     deq = deque([p])
    #     while deq:
    #         p = deq.popleft()
    #         if p in ps and p not in visited:
    #             visited.add(p)
    #             s.add(p)
    #             deq.extend(p.neighbors8)
    #     return s

    @staticmethod
    def bfs_array_to_points2(p: Point2, array: np.ndarray) -> Set[Point2]:
        s = set()
        p = p.rounded
        deq = deque([(p.x, p.y)])
        while deq:
            x, y = deq.popleft()
            if array[y, x]:
                p = Point2((x, y))
                s.add(p)
                deq.extend(p.neighbors8)
                array[y, x] = False
        return s

    @staticmethod
    def bfs_region(center: Point2, array: np.ndarray) -> np.ndarray:
        center = center.rounded
        deq = deque([(center.x, center.y)])
        region_points = np.zeros(array.shape, dtype=bool)
        while deq:
            x, y = deq.popleft()
            if array[y, x]:
                if array[y, x] == 1:
                    deq.extend(Point2((x, y)).neighbors4)
                array[y, x] = False
                region_points[y, x] = True
        return region_points

    def get_numpy_terrain_height(self, p: Point2):
        x, y = p.rounded
        return self.terrain_height[x, y]

    def plot_regions(self):
        placement = np.zeros(self.map_shape)
        for val, reg in self.regions.items():
            placement[reg.array] = val
        plt.matshow(placement)
        plt.show()

    def draw_connectivity_lines(self, include_points=False):
        visited: Set[Region] = set()
        for reg in self.regions.values():
            if reg not in visited:
                visited.add(reg)
                for neighbour, connects in reg.connectivity_dict.items():
                    if neighbour not in visited:
                        for connect in connects:
                            if include_points:
                                for p in connect.points:
                                    h = self.get_terrain_z_height(p) + 0.3
                                    self._client.debug_box2_out(Point3((*p, h)) + Point2((0.5, 0.5)), 0.25, (255, 0, 0))
                                for p in connect.neighbour_side.points:
                                    h = self.get_terrain_z_height(p) + 0.3
                                    self._client.debug_box2_out(Point3((*p, h)) + Point2((0.5, 0.5)), 0.25, (0, 0, 255))
                            # p1, p2 = connect.edge_line
                            # h1 = self.get_terrain_z_height(p1) + 0.3
                            # h2 = self.get_terrain_z_height(p2) + 0.3
                            # self._client.debug_sphere_out(Point3((*p1, h1)) + Point2((0.5, 0.5)), r=0.5,
                            #                               color=(255, 0, 0))
                            # self._client.debug_sphere_out(Point3((*p2, h2)) + Point2((0.5, 0.5)), r=0.5,
                            #                               color=(255, 0, 0))
                            # self._client.debug_line_out(Point3((*p1, h1)) + Point2((0.5, 0.5)),
                            #                             Point3((*p2, h2)) + Point2((0.5, 0.5)))
                            # p1, p2 = connect.neighbour_side.edge_line
                            # h1 = self.get_terrain_z_height(p1) + 0.35
                            # h2 = self.get_terrain_z_height(p2) + 0.35
                            # self._client.debug_sphere_out(Point3((*p1, h1)) + Point2((0.5, 0.5)), r=0.5,
                            #                               color=(0, 0, 255))
                            # self._client.debug_sphere_out(Point3((*p2, h2)) + Point2((0.5, 0.5)), r=0.5,
                            #                               color=(0, 0, 255))
                            # self._client.debug_line_out(Point3((*p1, h1)) + Point2((0.5, 0.5)),
                            #                             Point3((*p2, h2)) + Point2((0.5, 0.5)))
                            p1 = connect.center
                            p2 = connect.neighbour_side.center
                            h1 = self.get_terrain_z_height(p1) + 0.3
                            h2 = self.get_terrain_z_height(p2) + 0.3
                            p1 = Point3((*p1, h1)) + Point2((0.5, 0.5))
                            p2 = Point3((*p2, h2)) + Point2((0.5, 0.5))
                            self._client.debug_sphere_out(p1, r=0.5)
                            self._client.debug_sphere_out(p2, r=0.5)
                            self._client.debug_line_out(p1, p2)

    def draw_ramp_points(self):
        for ramp in self.ramps:
            for p in ramp.points:
                h = self.get_terrain_z_height(p)
                self._client.debug_box2_out(Point3((*p, h + 0.2)) + Point2((0.5, 0.5)), 0.25, (255, 255, 255))
            for p in ramp.upper:
                h = self.get_terrain_z_height(p)
                self._client.debug_box2_out(Point3((*p, h + 0.2)) + Point2((0.5, 0.5)), 0.25, (0, 0, 255))
            for p in ramp.lower:
                h = self.get_terrain_z_height(p)
                self._client.debug_box2_out(Point3((*p, h + 0.2)) + Point2((0.5, 0.5)), 0.25, (255, 0, 0))
            p = ramp.upper_side_point()
            h = self.get_terrain_z_height(p)
            self._client.debug_sphere_out(Point3((*p, h + 0.1)) + Point2((0.5, 0.5)), r=0.5)
            p = ramp.lower_side_point()
            h = self.get_terrain_z_height(p)
            self._client.debug_sphere_out(Point3((*p, h + 0.1)) + Point2((0.5, 0.5)), r=0.5)

    def draw_regions_perimeter(self, up: float = 0.25):
        for reg in self.regions.values():
            color = (0, 255, 0)
            for p in reg.perimeter_as_points2:
                h = self.get_terrain_z_height(p)
                self._client.debug_box2_out(Point3((*p, h + up)) + Point2((0.5, 0.5)), 0.25, color=color)

    @staticmethod
    def plot_depth(points, graph: Dict[Point2, List[Point2]], chokes, center):
        from matplotlib.patches import ConnectionPatch
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        coordsA = "data"
        for items in graph.items():
            if len(items[1]) > 1:
                p = items[0]
                for n in items[1][1:]:
                    ax.add_patch(ConnectionPatch(p, n, coordsA, coordsA, arrowstyle="->", shrinkB=5, shrinkA=5))
        for choke in chokes:
            ax.add_patch(ConnectionPatch(choke[0], choke[1], coordsA, coordsA, color='#ff1100'))

        plt.scatter(*zip(*points))
        plt.scatter(*center, c='#800080', marker='^')
        plt.show()

    def get_terrain_z_height(self, pos: Union[Point2, Unit]) -> float:
        """Returns terrain z-height at a position.

        :param pos:"""
        assert isinstance(pos, (Point2, Unit)), f"pos is not of type Point2 or Unit"
        pos = pos.position.rounded
        return -16 + 32 * self._game_info.terrain_height[pos] / 255

from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Set, Optional, DefaultDict

from sc2.bot_ai import BotAI
from sc2.units import Units
from sc2.unit import Unit
from sc2.position import Point2, Point3
import matplotlib.pyplot as plt
from skimage.draw import line
import numpy as np
import math

from MapRegions.Region import Region, Polygon
from MapRegions.constructs import Blocker, Passage
from MapRegions.connectivity import ConnectivitySide
from MapRegions.MapGrid import MapGrid
from MapRegions.utils import rotation_matrix, get_outliers


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
        self.expansion_locations = []
        for loc in list(self.bot.expansion_locations_dict.keys()):
            if self.bot.start_location.is_same_as(loc, 1):
                self.bot_start_loc = loc
            elif self.bot.enemy_start_locations[0].is_same_as(loc, 1):
                self.enemy_start_loc = loc
            self.expansion_locations.append(loc)

        # numpy grids
        self.placement_grid = self._game_info.placement_grid.data_numpy.copy()
        self.pathing_grid = self._game_info.pathing_grid.data_numpy.copy()
        self.map_shape = self.pathing_grid.shape

        self.passages = self._find_passages()

        self._regions: Dict[int, Region] = self.slice_map()
        self._regions_map = np.zeros(self.map_shape)
        for reg in self._regions.values():
            self._regions_map[reg.ndarray] = reg.label
        self._connect_regions()

    def region_at(self, pos: Union[Unit, Point2]) -> Region:
        if isinstance(pos, Unit):
            pos = pos.position
        idx = self._regions_map[pos.y, pos.x]
        return self._regions[idx]

    @property
    def regions(self) -> Dict[int, Region]:
        return self._regions

    @property
    def regions_map(self) -> np.ndarray:
        return self._regions_map

    def slice_map(self) -> Dict[int, Region]:
        def add_chokes(chokes, map_grid: MapGrid):
            for choke in chokes:
                x1, y1 = choke[0][1], choke[0][0]
                x2, y2 = choke[1][1], choke[1][0]
                rr, cc = line(x1, y1, x2, y2)
                for x, y in zip(rr, cc):
                    if map_grid.grid[x, y]:
                        map_grid.grid[x, y] = 2

        regions = []
        map_center = self._game_info.map_center
        region_grid = self.region_grid(MapGrid(self.placement_grid), 2, True)

        # calculate bases locations
        for loc in self.expansion_locations:
            if not (loc == self.bot_start_loc or loc == self.enemy_start_loc):
                add_chokes(self.find_chokes(loc, distance=24), region_grid)
            regions.append(Polygon(region_grid.bfs_ndarray(loc, rng=1, sub=True)))

        # side points of ramps, blockers
        side_points = []
        for passage in self.passages:
            groups = passage.surrounding_groups
            if len(groups) == 2:
                c1, c2 = Point2.center(groups[0]), Point2.center(groups[1])
                vec1, vec2 = c1.direction_vector(c2).normalized, c2.direction_vector(c1).normalized
                side_points.extend([6 * vec1 + c2, 6 * vec2 + c1])

        # sort side points by distance to reflection point/line
        basses_center = Point2.center([self.bot_start_loc, self.enemy_start_loc])
        if basses_center.is_same_as(map_center, 1):
            side_points = map_center.sort_by_distance(side_points)
        else:
            side_points = sorted(side_points, key=lambda p: np.linalg.norm(
                np.cross(basses_center - map_center, map_center - p)) / np.linalg.norm(
                basses_center - map_center))

        # calculate side points of ramps and blockers
        for point in side_points:
            if region_grid.value_at(point):
                add_chokes(self.find_chokes(point), region_grid)
                regions.append(Polygon(region_grid.bfs_ndarray(point, rng=1, sub=True)))

        # rest points
        for (b, a), value in np.ndenumerate(region_grid.grid):
            p = Point2((a, b))
            if value > 0:
                regions.append(Polygon(region_grid.bfs_ndarray(p, rng=1, sub=True)))

        # connecting smaller regions into bigger
        regions = sorted(self._clean_regions(regions), key=lambda x: self.bot_start_loc.distance_to_point2(x.center))

        # labeling regions
        for idx, reg in enumerate(regions):
            reg.label = idx + 1
        return {reg.label: reg for reg in regions}

    @staticmethod
    def _region_neighbours_label(reg: Polygon, grid: np.ndarray):
        neighbours_set = set()
        indices = np.nonzero(reg.ndarray)
        x, y = indices[0][0], indices[1][0]
        val = grid[x, y]
        deq = [Point2((y, x))]
        region_points = set()
        # BFS region points
        while deq:
            p: Point2 = deq.pop(0)
            if p not in region_points:
                region_points.add(p)
                if grid[p.y][p.x] == val:
                    deq.extend(p.neighbors8)
                elif grid[p.y][p.x] > 0:
                    neighbours_set.add(grid[p.y][p.x])
        return neighbours_set

    def _clean_regions(self, regions: List[Polygon], smallest_area_limit: int = 100) -> List[Region]:
        # preparing the data
        regions_grid = MapGrid(np.zeros(self.map_shape))
        regions_dict: Dict[int, Polygon] = dict()
        for idx, reg in enumerate(regions):
            reg.label = idx + 1
            regions_dict[reg.label] = reg
            regions_grid.include_grid(reg.ndarray, status=reg.label)

        # func output
        regs: List[Region] = []

        regions = sorted(regions, key=lambda p: p.area)
        for reg in regions:
            if reg.area >= smallest_area_limit:
                continue
            # trying to add regions which are below area limit to their neighbours
            # neighbour_regs = {
            #     regions_dict[regions_grid.value_at(p)] for p in
            #     regions_grid.bfs_points(reg.first_point, rng=reg.label, neighbour8=True)[1]
            #     if regions_grid.value_at(p)
            # }
            neighbour_regs = {
                regions_dict[v] for v in self._region_neighbours_label(reg, regions_grid.grid)
            }

            # filter if there is neighbour without base and is above size limit
            non_base_regs = {
                neighbour
                for neighbour in neighbour_regs
                if not any(neighbour.is_inside_indices(exp) for exp in self.expansion_locations)
            }

            if non_base_regs:
                neighbour_regs = non_base_regs

            # search for smallest neighbour (prefer region above area limit)
            # leave region if there isn't neighbour without base and himself is big enough
            if neighbour_regs is not None and reg.area < smallest_area_limit // 2:
                smallest_reg: Polygon = min([reg for reg in neighbour_regs if reg.area >= smallest_area_limit],
                                            key=lambda x: x.area, default=None)
                if not smallest_reg:
                    smallest_reg: Polygon = max([reg for reg in neighbour_regs if reg.area < smallest_area_limit],
                                                key=lambda x: x.area, default=None)
                if smallest_reg:
                    # add reg data to found smallest neighbour
                    regions_dict.pop(reg.label)
                    regions_dict[smallest_reg.label].ndarray[reg.ndarray] = True
                    regions_grid.grid[reg.ndarray] = smallest_reg.label

        for key, item in regions_dict.items():
            ndarray = np.zeros(self.map_shape, dtype=bool)
            ndarray[np.nonzero(item.ndarray)] = True
            bases = [base for base in self.expansion_locations if item.is_inside_indices(base)]
            watchtowers = self.watchtowers.filter(lambda x: item.is_inside_indices(x.position))
            regs.append(Region(ndarray, bases, watchtowers))
        return regs

    def _connect_regions(self):
        # creating a perimeter grid of all regions
        regions_perimeter_grid = np.zeros(self.map_shape, dtype=np.uint8)
        for val, reg in self._regions.items():
            regions_perimeter_grid[reg.perimeter_as_indices] = val
        regions_perimeter_grid = regions_perimeter_grid.T

        for w, group in enumerate(self.passages):
            surrounding_groups = group.surrounding_groups
            if len(surrounding_groups) == 2:
                first_side = [0] * len(self._regions)
                second_side = [0] * len(self._regions)
                for p in surrounding_groups[0]:
                    val = int(self._regions_map[p.y, p.x])
                    first_side[val - 1] = first_side[val - 1] + 1
                for p in surrounding_groups[1]:
                    val = int(self._regions_map[p.y, p.x])
                    second_side[val - 1] = second_side[val - 1] + 1
                for idx1, l1 in enumerate(first_side):
                    for idx2, l2 in enumerate(second_side):
                        if l1 >= len(surrounding_groups[0]) * 0.4 and l2 >= len(surrounding_groups[1]) * 0.4:
                            self.add_connectivity(self._regions[idx1 + 1], self._regions[idx2 + 1], surrounding_groups[0],
                                                  surrounding_groups[1], False, False)

        for val, reg in self._regions.items():
            d: Dict[int, set] = {v: set() for v in self._regions.keys()}
            to_del = set()
            for p in reg.perimeter_as_points2:
                for n in p.neighbors8:
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
                        neighbour = self._regions[v]
                        g1 = {p for p in group if reg.is_inside_point(p)}
                        g2 = {p for p in group if neighbour.is_inside_point(p)}
                        self.add_connectivity(reg, neighbour, g1, g2, False, False)

    def add_connectivity(self, reg_a: Region, reg_b: Region,
                         points_a: Set[Point2], points_b: Set[Point2],
                         impassable: bool, jumpable: bool,
                         blocker: Optional[Blocker] = None):
        def add(r: Region, c: ConnectivitySide):
            r.connectivity_dict[c.neighbour.label].append(c)

        connect_a = ConnectivitySide(self, reg_a, None, points_a, impassable, jumpable, blocker)
        connect_b = ConnectivitySide(self, reg_b, connect_a, points_b, impassable, jumpable, blocker)
        connect_a.neighbour_side = connect_b

        add(reg_a, connect_a)
        add(reg_b, connect_b)

    def region_grid(self, map_grid: MapGrid, val: int, rich=False) -> MapGrid:

        # destructables
        map_grid.include_destructables(self.destructables, False)

        # mineral blockers
        if rich:
            map_grid.include_destructables(self.mineral_blockers, False)
        else:
            map_grid.include_destructables(self.mineral_blockers.filter(lambda u: "450" in u.name.lower()), False)

        # vision blockers
        map_grid.include_points(self._game_info.vision_blockers, val)

        return map_grid

    def fixed_pathing_grid(self):
        map_grid = MapGrid(self.pathing_grid)
        map_grid.include_destructables(self.destructables | self.mineral_blockers, True)
        return map_grid.grid

    def _find_passages(self) -> List[Passage]:
        pathing_map = MapGrid(self.pathing_grid)
        pathing_map.include_grid(self.placement_grid, True)
        pathing_map.include_destructables(self.destructables | self.mineral_blockers, True)
        placement_map = MapGrid(self.placement_grid)
        placement_map.include_points(self._game_info.vision_blockers, True)
        placement_map.include_destructables(self.destructables | self.mineral_blockers, False)
        pathing_map.include_grid(placement_map.grid, False)

        passages: List[Passage] = []

        for (a, b), value in np.ndenumerate(pathing_map.grid):
            if value:
                ps, sur = pathing_map.bfs_points(Point2((b, a)), sub=True, neighbour8=True)
                if len(ps) > 8:
                    blockers = (self.mineral_blockers | self.destructables).filter(lambda x: x.position.rounded in ps)
                    passages.append(Passage(ps, sur, blockers, self._game_info))

        return passages

    def depth_points(self, center: Point2, dis, degrees_step) -> List[Point2]:
        grid = MapGrid(self.pathing_grid)
        grid.include_grid(self.placement_grid, True)
        grid = self.region_grid(grid, 0).grid.T
        vec = center.direction_vector(self._game_info.map_center).normalized
        vec_arr = np.array([[vec.x], [vec.y]])
        point_list = []
        for deg in range(0, 360, degrees_step):
            x, y = np.matmul(rotation_matrix(deg), vec_arr).round(5)
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
        data = get_outliers(np.array(distance))[0]
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

    def plot_regions(self):
        placement = np.zeros(self.map_shape)
        for val, reg in self._regions.items():
            placement[reg.ndarray] = val
        plt.matshow(placement)
        plt.show()

    def draw_connectivity_lines(self, include_points=False):
        visited: Set[Region] = set()
        for reg in self._regions.values():
            if reg not in visited:
                visited.add(reg)
                for neighbour, connects in reg.connectivity_dict.items():
                    if neighbour not in visited:
                        for connect in connects:
                            if include_points:
                                for p in connect.points:
                                    h = self.get_terrain_z_height(p) - 0.3
                                    self._client.debug_box2_out(Point3((*p, h)) + Point2((0.5, 0.5)), 0.4, (255, 0, 0))
                                for p in connect.neighbour_side.points:
                                    h = self.get_terrain_z_height(p) - 0.3
                                    self._client.debug_box2_out(Point3((*p, h)) + Point2((0.5, 0.5)), 0.4, (0, 0, 255))
                            p1 = connect.center
                            p2 = connect.neighbour_side.center
                            h1 = self.get_terrain_z_height(p1) + 0.3
                            h2 = self.get_terrain_z_height(p2) + 0.3
                            p1 = Point3((*p1, h1)) + Point2((0.5, 0.5))
                            p2 = Point3((*p2, h2)) + Point2((0.5, 0.5))
                            self._client.debug_sphere_out(p1, r=0.5)
                            self._client.debug_sphere_out(p2, r=0.5)
                            self._client.debug_line_out(p1, p2)

    def draw_regions_perimeter(self, up: float = 0.25):
        for reg in self._regions.values():
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

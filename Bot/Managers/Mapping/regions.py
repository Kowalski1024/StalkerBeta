from typing import List, Union, Dict, Set, NamedTuple
from dataclasses import dataclass

from sc2.game_info import GameInfo
from sc2.client import Client
from sc2.units import Units
from sc2.unit import Unit
from .constructs import Ramp
from sc2.position import Point2, Point3
from sc2.cache import property_immutable_cache, property_mutable_cache
import sc2math
import matplotlib.pyplot as plt
from skimage.draw import line
from collections import deque
import numpy as np
import math
import random

from .polygon import Polygon, BaseRegion


class Regions:
    def __init__(self,
                 game_info: GameInfo,
                 client: Client,
                 destructables: Units,
                 mineral_fields: Units,
                 expansion_locations,
                 bot_start_loc,
                 enemy_start_loc
                 ):
        self.cache = {}

        # game information
        self._game_info = game_info
        self._client = client
        self.destructables = destructables
        self.mineral_fields = mineral_fields
        self.expansion_locations = expansion_locations
        self.bot_start_loc = bot_start_loc
        self.enemy_start_loc = enemy_start_loc

        # numpy grids
        self.terrain_height = self._game_info.terrain_height.data_numpy
        self.placement_grid = self._game_info.placement_grid.data_numpy
        self.pathing_grid = self._game_info.pathing_grid.data_numpy
        self.map_shape = self.pathing_grid.shape

        self.ramps = self._find_ramps()
        self.polygons: List[Polygon] = self._slice_map()

        self.plot_polygons()

    def _slice_map(self):
        polygons = []
        map_center = self._game_info.map_center
        region_grid = self.region_grid(self.placement_grid.copy(), 2, True)

        # bases locations
        for loc in self.expansion_locations:
            if not (self.bot_start_loc.is_same_as(loc, 1) or self.enemy_start_loc.is_same_as(loc, 1)):
                self.add_chokes(self.find_chokes(loc, distance=24), region_grid)
            poly = BaseRegion(self.dfs_region(loc.rounded, region_grid), loc)
            polygons.append(poly)
        ramps_points = []
        for ramp in self.ramps:
            ramps_points.append(ramp.upper_side_point())
            ramps_points.append(ramp.down_side_point())

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
                poly = Polygon(self.dfs_region(point.rounded, region_grid))
                polygons.append(poly)

        region_grid[region_grid == 2] = 1
        for (b, a), value in np.ndenumerate(region_grid):
            p = Point2((a, b))
            if value > 0 and sc2math.point2_in_grid(p, region_grid):
                polygons.append(Polygon(self.dfs_region(p.rounded, region_grid)))
        return self._clean_polygons(polygons)

    def _clean_polygons(self, polygons, smallest_area: int = 100):
        def neighbours(_poly: Polygon, _grid: np.ndarray):
            neighbours_set = set()
            p = list(_poly.points)[0]
            val = _grid[p.y][p.x]
            deq = deque([p])
            polygon_points = np.zeros(_grid.shape, dtype=bool)
            while deq:
                p: Point2 = deq.popleft()
                if not polygon_points[p.y][p.x]:
                    polygon_points[p.y][p.x] = True
                    if _grid[p.y][p.x] == val:
                        deq.extend(p.neighbors8)
                    elif _grid[p.y][p.x] > 0:
                        neighbours_set.add(_grid[p.y][p.x])
            return neighbours_set

        map_size = self._game_info.placement_grid.data_numpy.shape
        polygons_grid = np.zeros(map_size)
        polygons_dict: Dict[int, Polygon] = dict()
        for idx, poly in enumerate(polygons):
            polygons_dict[idx + 1] = poly
            polygons_grid[poly.array] = idx + 1

        polys = []
        polygons = sorted(polygons, key=lambda p: p.area)
        for poly in polygons:
            if poly.area < smallest_area:
                neighbour_polys: List[Polygon] = [
                    polygons_dict[v] for v in neighbours(poly, polygons_grid)
                ]
                non_base_polys = [
                    neighbour for neighbour in neighbour_polys
                    if not neighbour.has_base and neighbour.area >= smallest_area
                ]
                if non_base_polys:
                    neighbour_polys = non_base_polys
                elif poly.area >= 50:
                    polys.append(poly)
                    continue
                above = [neighbour for neighbour in neighbour_polys if neighbour.area >= smallest_area]
                under = [neighbour for neighbour in neighbour_polys if neighbour.area < smallest_area]

                if above:
                    smallest_poly = min(above, key=lambda x: x.area)
                else:
                    if not under:
                        polys.append(poly)
                        continue
                    smallest_poly = min(under, key=lambda x: x.area)
                smallest_poly.add_polygon(poly)
            else:
                polys.append(poly)
        return polys

    def region_grid(self, grid: np.ndarray, val: int, rich=False) -> np.ndarray:
        from MapAnalyzer.utils import change_destructable_status_in_grid

        # destructables
        constructs_grid = np.zeros(self.map_shape, dtype=bool).T
        for unit in self.destructables:
            change_destructable_status_in_grid(constructs_grid, unit, status=True)
        grid[constructs_grid.T] = False
        # for x, y in sc2math.ndarray_corners_points2(constructs_grid.T):
        #     if grid[y, x]:
        #         grid[y, x] = val

        # mineral blockers
        resource_blockers = [m.position for m in self.mineral_fields if "450" in m.name.lower()]
        if rich:
            resource_blockers.extend([m.position for m in self.mineral_fields if "rich" in m.name.lower()])
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
        heights = {self.get_numpy_terrain_height(p) for p in self.expansion_locations}
        _min, _max = min(heights), max(heights)

        points = {
            Point2((b, a))
            for a in range(self.map_shape[0]) for b in range(self.map_shape[1])
            if _min < self.terrain_height[a, b] < _max
               and self.terrain_height[a, b] not in heights
        }
        ramps = [Ramp(group, self._game_info) for group in self._game_info._find_groups(points)]
        return [ramp for ramp in ramps if len(ramp.heights()) >= 4]

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
            graph[p] = [(ps[(idx+1) % len(ps)])]

        for idx, p in enumerate(ps):
            prev_p = ps[idx-1]
            next_p = ps[(idx+1) % len(ps)]
            center_dist = p.distance_to_point2(center)
            prev_dist = p.distance_to_point2(prev_p)
            next_dist = p.distance_to_point2(next_p)
            ll = math.radians(degrees_step)*center_dist*2
            if prev_dist >= ll + 1:
                for i in range(2, int(len(ps)/4)):
                    dis = p.distance_to_point2(ps[idx-i])
                    if dis <= prev_dist:
                        prev_dist = dis
                        prev_p = ps[idx-i]
                graph[prev_p].append(p)

            if next_dist >= ll + 1:
                for i in range(2, int(len(ps)/4)):
                    dis = p.distance_to_point2(ps[(idx+i) % len(ps)])
                    if dis <= next_dist:
                        next_dist = dis
                        next_p = ps[(idx+i) % len(ps)]
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

        # self.plot_depth(ps, graph, chokes, center)
        return chokes

        # distance = [p1.distance_to_point2(p2) for p1, p2 in zip(points, points[1:] + points[:1])]
        # distance_to_center = [p.distance_to_point2(center) for p in points]
        # rotate = distance_to_center.index(min(distance_to_center))
        # points = deque(points)
        # distance = deque(distance)
        # distance_to_center = deque(distance_to_center)
        # points.rotate(-rotate)
        # distance.rotate(-rotate)
        # distance_to_center.rotate(-rotate)
        # points = list(points)
        # distance_to_center = list(distance_to_center)
        # points_length = len(points)
        # chokes = []
        # _next = 0
        # deq = deque(points[-int(points_length / 4):], maxlen=int(points_length / 4))
        # for idx in range(points_length):
        #     if idx < _next:
        #         continue
        #     p = points[idx]
        #     d = distance_to_center[idx]
        #     deq.append(p)
        #     if distance[idx] >= math.radians(degrees_step) * d * 2 + math.sqrt(1.6):
        #         _min = 100
        #         _next = 0
        #         closest_points = (points[idx + 1:] + points)[:int(points_length / 4)]
        #         side_a = p
        #         side_b = points[idx + 1] if idx + 1 < points_length else points[0]
        #         for point, it in zip(closest_points, range(int(points_length / 4))):
        #             dis = math.hypot(p[0] - point[0], p[1] - point[1])
        #             if dis < _min:
        #                 _next = it + 1
        #                 _min = dis
        #                 side_b = point
        #         _next += idx
        #         _min = 100
        #         for point in deq:
        #             dis = math.hypot(side_b[0] - point[0], side_b[1] - point[1])
        #             if dis < _min:
        #                 _min = dis
        #                 side_a = point
        #         chokes.append([side_a, side_b])
        # loc_chokes = clean_chokes(chokes)
        # # debug plot
        # # self.plot_depth(points, loc_chokes, center)
        # return loc_chokes

    @staticmethod
    def add_chokes(chokes, grid: np.ndarray):
        for choke in chokes:
            x1, y1 = choke[0][1], choke[0][0]
            x2, y2 = choke[1][1], choke[1][0]
            rr, cc = line(x1, y1, x2, y2)
            for x, y in zip(rr, cc):
                if grid[x, y]:
                    grid[x, y] = 2

    @staticmethod
    def dfs_region(center: Point2, array: np.ndarray) -> np.ndarray:
        center = center.rounded
        deq = deque([(center.x, center.y)])
        polygon_points = np.zeros(array.shape, dtype=bool)
        while deq:
            x, y = deq.popleft()
            if array[y, x]:
                if array[y, x] == 1:
                    deq.extend(Point2((x, y)).neighbors4)
                array[y, x] = False
                polygon_points[y, x] = True
        return polygon_points

    def get_numpy_terrain_height(self, p: Point2):
        x, y = p.rounded
        return self.terrain_height[x, y]

    def plot_polygons(self):
        placement = np.zeros(self.map_shape)
        it = 1
        for poly in self.polygons:
            placement[poly.array] = it
            it += 1
        plt.matshow(placement)
        plt.show()

    def draw_divider_centers(self):
        for ramp in self.ramps:
            top_center, center, down_center = ramp.centers
            vec = (down_center - center).normalized
            down = 8 * vec + center
            top = -8 * vec + center
            h1 = self.get_terrain_z_height(down)
            h2 = self.get_terrain_z_height(top)
            self._client.debug_sphere_out(Point3((*down, h1)), r=1)
            self._client.debug_sphere_out(Point3((*top, h2)), r=1)
        for loc in self.expansion_locations:
            h = self.get_terrain_z_height(loc)
            self._client.debug_sphere_out(Point3((*loc, h)), r=1)

    def draw_ramp_points(self):
        for ramp in self.ramps:
            height_points = ramp.heights()
            mid = len(height_points) / 2
            top_height = height_points[int(mid + 1)]
            bottom_height = height_points[int(mid - 1)]
            upper = []
            down = []
            for p in ramp.points:
                h = self.get_terrain_z_height(p)
                if h == top_height:
                    upper.append(p)
                    self._client.debug_box2_out(Point3((*p, h + 0.25)), 0.25, Point3((255, 0, 0)))
                elif h == bottom_height:
                    down.append(p)
                    self._client.debug_box2_out(Point3((*p, h + 0.25)), 0.25, Point3((0, 255, 0)))
                else:
                    self._client.debug_box2_out(Point3((*p, h + 0.25)), 0.25)
            u, d = Point2.center(upper), Point2.center(down)
            h1 = self.get_terrain_z_height(u)
            h2 = self.get_terrain_z_height(d)
            self._client.debug_box2_out(Point3((*u, h1 + 0.25)), 0.25, Point3((0, 0, 255)))
            self._client.debug_box2_out(Point3((*d, h2 + 0.25)), 0.25, Point3((0, 0, 255)))

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

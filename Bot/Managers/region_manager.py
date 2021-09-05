
from .manager_base import ManagerBase
from sc2.game_info import GameInfo
from .Mapping.regions import Regions


# class ConnectivityLine:
#     def __init__(self, parent: 'Polygon', neighbour: 'Polygon'):
#         self.parent = parent
#         self.neighbour = neighbour
#         self.points: Set[Point2] = set()
#
#
# class Polygon:
#     def __init__(self, arr: np.ndarray):
#         self.has_base = False
#         self.array = arr
#         self.points = {Point2((y, x)) for x, y in zip(*np.nonzero(arr))}
#         self.connectivity_dict: Dict[Polygon, List[ConnectivityLine]] = dict()
#
#     def add_point(self, p: Union[Point2, tuple]):
#         self.array[p[1]][p[0]] = True
#         self.points.add(p)
#
#     def add_points(self, points: Union[set, list]):
#         for p in points:
#             self.add_point(p)
#
#     def add_polygon(self, poly: 'Polygon'):
#         self.array[poly.array] = True
#         self.points.update(poly.points)
#
#     def connect_polygon(self, connect_line: ConnectivityLine, no_exists=False):
#         if connect_line.parent != self:
#             connect_line.neighbour, connect_line.parent = connect_line.parent, connect_line.neighbour
#         if no_exists and connect_line.neighbour in self.connectivity_dict:
#             return
#         if connect_line.neighbour not in self.connectivity_dict:
#             self.connectivity_dict[connect_line.neighbour] = [connect_line]
#         else:
#             self.connectivity_dict[connect_line.neighbour].append(connect_line)
#
#     @property
#     def area(self) -> int:
#         return len(self.points)
#
#
# class BaseRegion(Polygon):
#     def __init__(self, arr: np.ndarray, base: Point2):
#         super().__init__(arr)
#         self.base_location = base
#         self.has_base = True


# class Ramp:
#     def __init__(self, points: Set[Point2], game_info: GameInfo):
#         self._points = points
#         self.__game_info = game_info
#
#         self.cache = {}
#
#     @property_immutable_cache
#     def _height_map(self):
#         return self.__game_info.terrain_height
#
#     def height_at(self, p: Point2) -> int:
#         return self._height_map[p]
#
#     def heights(self):
#         return sorted(list({self.height_at(point) for point in self._points}))
#
#     @property_mutable_cache
#     def points(self) -> Set[Point2]:
#         return self._points.copy()
#
#     @property_mutable_cache
#     def centers(self) -> List[Point2]:
#         height_points = self.heights()
#         mid = len(height_points) / 2
#         if len(height_points) > 2:
#             top_height = height_points[int(mid + 1)]
#             bottom_height = height_points[int(mid - 1)]
#         else:
#             top_height = height_points[0]
#             bottom_height = height_points[1]
#         upper = []
#         down = []
#         for p in self._points:
#             h = self.height_at(p)
#             if h == top_height:
#                 upper.append(p)
#             elif h == bottom_height:
#                 down.append(p)
#         u, d = Point2.center(upper), Point2.center(down)
#         return [u, Point2.center([u, d]), d]
#
#     def upper_side_point(self, distance=8):
#         top_center, center, down_center = self.centers
#         vec = (down_center - center).normalized
#         return -distance * vec + center
#
#     def down_side_point(self, distance=8):
#         top_center, center, down_center = self.centers
#         vec = (down_center - center).normalized
#         return distance * vec + center


class RegionManager(ManagerBase):
    def __init__(self, bot):
        super().__init__()
        self._bot = bot
        self._game_info: GameInfo = bot.game_info
        expansion_locations = list(self._bot.expansion_locations_dict.keys())
        self.regions = Regions(
            self._bot.game_info,
            self._bot.client,
            self._bot.destructables,
            self._bot.mineral_field,
            expansion_locations,
            self._bot.start_location,
            self._bot.enemy_start_locations[0]
        )



    # def _find_ramps(self):
    #     heights = {self.get_numpy_terrain_height(p) for p in self.expansion_locations}
    #     _min, _max = min(heights), max(heights)
    #
    #     points = {
    #         Point2((a, b))
    #         for (b, a), value in np.ndenumerate(self._game_info.pathing_grid.data_numpy)
    #         if _min < self.terrain_height[b, a] < _max
    #            and self.terrain_height[b, a] not in heights
    #     }
    #     ramps = [Ramp(group, self._game_info) for group in self._game_info._find_groups(points)]
    #     return [ramp for ramp in ramps if len(ramp.heights()) >= 4]
    #
    # def get_numpy_terrain_height(self, p: Point2):
    #     x, y = p.rounded
    #     return self._game_info.terrain_height.data_numpy[y, x]
    #
    # def _clean_polygons(self, polygons, smallest_area: int = 100):
    #     def neighbours(_poly: Polygon, _grid: np.ndarray):
    #         neighbours_set = set()
    #         p = random.sample(_poly.points, 1)[0]
    #         val = _grid[p.y][p.x]
    #         deq = deque([p])
    #         polygon_points = np.zeros(_grid.shape, dtype=bool)
    #         while deq:
    #             p: Point2 = deq.popleft()
    #             if not polygon_points[p.y][p.x]:
    #                 polygon_points[p.y][p.x] = True
    #                 if _grid[p.y][p.x] == val:
    #                     deq.extend(p.neighbors8)
    #                 elif _grid[p.y][p.x] > 0:
    #                     neighbours_set.add(_grid[p.y][p.x])
    #         return neighbours_set
    #
    #     map_size = self._game_info.placement_grid.data_numpy.shape
    #     polygons_grid = np.zeros(map_size)
    #     polygons_dict: Dict[int, Polygon] = dict()
    #     for idx, poly in enumerate(polygons):
    #         polygons_dict[idx+1] = poly
    #         polygons_grid[poly.array] = idx+1
    #
    #     polys = []
    #     polygons = sorted(polygons, key=lambda p: p.area)
    #     for poly in polygons:
    #         if poly.area < smallest_area:
    #             neighbour_polys: List[Polygon] = [
    #                 polygons_dict[v] for v in neighbours(poly, polygons_grid)
    #             ]
    #             non_base_polys = [
    #                 neighbour for neighbour in neighbour_polys
    #                 if not neighbour.has_base and neighbour.area >= smallest_area
    #             ]
    #             if non_base_polys:
    #                 neighbour_polys = non_base_polys
    #             elif poly.area >= 50:
    #                 polys.append(poly)
    #                 continue
    #             above = [neighbour for neighbour in neighbour_polys if neighbour.area >= smallest_area]
    #             under = [neighbour for neighbour in neighbour_polys if neighbour.area < smallest_area]
    #
    #             if above:
    #                 smallest_poly = min(above, key=lambda x: x.area)
    #             else:
    #                 if not under:
    #                     polys.append(poly)
    #                     continue
    #                 smallest_poly = min(under, key=lambda x: x.area)
    #             smallest_poly.add_polygon(poly)
    #         else:
    #             polys.append(poly)
    #     return polys
    #
    # @staticmethod
    # def sub_chokes_from_grid(chokes, grid: np.ndarray):
    #     for choke in chokes:
    #         x1, y1 = choke[0][1], choke[0][0]
    #         x2, y2 = choke[1][1], choke[1][0]
    #         rr, cc = line(x1, y1, x2, y2)
    #         for x, y in zip(rr, cc):
    #             if grid[(x, y)]:
    #                 grid[(x, y)] = 2
    #
    # def divide_map(self):
    #     map_center = self._game_info.map_center
    #     start_loc = self._bot.start_location
    #     enemy_loc = self._bot.enemy_start_locations[0]
    #     polygons = []
    #     placement_grid = self._game_info.placement_grid.data_numpy.copy()
    #     vb = sc2math.points2_to_indices(self._game_info.vision_blockers)
    #     placement_grid[vb] = 2
    #
    #     # bases locations
    #     for loc in self.expansion_locations:
    #         if not (start_loc.is_same_as(loc, 1) or enemy_loc.is_same_as(loc, 1)):
    #             self.sub_chokes_from_grid(self.find_chokes(loc), placement_grid)
    #         poly = BaseRegion(sc2math.dfs_numpy(loc.rounded, placement_grid), loc)
    #         polygons.append(poly)
    #
    #     ramps_points = []
    #     for ramp in self.map_ramps:
    #         ramps_points.append(ramp.upper_side_point())
    #         ramps_points.append(ramp.down_side_point())
    #
    #     # sort ramps by distance to reflection point/line
    #     if Point2.center([start_loc, enemy_loc]).is_same_as(map_center, 1):
    #         ramps_points = map_center.sort_by_distance(ramps_points)
    #     else:
    #         center = Point2.center([start_loc, enemy_loc])
    #         ramps_points = sorted(ramps_points, key=lambda p: np.linalg.norm(
    #             np.cross(center - map_center, map_center - p)) / np.linalg.norm(
    #             center - map_center))
    #
    #     for point in ramps_points:
    #         if sc2math.point2_in_grid(point, placement_grid):
    #             chokes = self.find_chokes(point)
    #             self.sub_chokes_from_grid(chokes, placement_grid)
    #             poly = Polygon(sc2math.dfs_numpy(point.rounded, placement_grid))
    #             polygons.append(poly)
    #
    #     for (b, a), value in np.ndenumerate(placement_grid):
    #         p = Point2((a, b))
    #         if value == 1 and sc2math.point2_in_grid(p, placement_grid):
    #             polygons.append(Polygon(sc2math.dfs_numpy(p.rounded, placement_grid)))
    #     return self._clean_polygons(polygons)
    #
    # # @staticmethod
    # # def create_polygon(center: Union[tuple, Point2], grid: np.ndarray) -> Polygon:
    # #     center = int(center[0]), int(center[1])
    # #     polygon_array = sc2_math.dfs_numpy(center, grid)
    # #     return Polygon(polygon_array)
    #
    # def depth_points(self, center: Union[tuple, Unit, Point2], dis, degrees_step) -> List[Point2]:
    #     if isinstance(center, Unit):
    #         center = center.position
    #     elif isinstance(center, tuple):
    #         center = Point2(center)
    #
    #     pathing_grid = self._bot.game_info.pathing_grid.data_numpy.copy()
    #     pathing_grid[np.nonzero(self._bot.game_info.placement_grid.data_numpy)] = True
    #     vb = sc2math.points2_to_indices(self._game_info.vision_blockers)
    #     pathing_grid[vb] = False
    #
    #     vec = np.array([[1], [0]])
    #     point_list = list()
    #     deg_list = range(0, 360, degrees_step)
    #     if center.distance_to(self._bot.start_location) > center.distance_to(self._bot.enemy_start_locations[0]):
    #         deg_list = reversed(deg_list)
    #     for deg in deg_list:
    #         x, y = np.matmul(sc2math.rotation_matrix(deg), vec).round(5)
    #         x, y = x[0], y[0]
    #         for m in range(1, dis):
    #             if not pathing_grid[int(x * m + center.y)][int(y * m + center.x)]:
    #                 point_list.append((int(y * m + center.x), int(x * m + center.y)))
    #                 break
    #
    #     # clean list from duplicates
    #     point_list = list(dict.fromkeys(point_list))
    #     # clean list from outliers
    #     distance = [math.hypot(p[0] - center.x, p[1] - center.y) for p in point_list]
    #     data = sc2math.get_outliers(np.array(distance))[0]
    #     data = {i for i in data if distance[i] > np.mean(np.array(distance))}
    #     return [Point2(i) for j, i in enumerate(point_list) if j not in data]
    #
    # def find_chokes(self, center: Union[tuple, Unit, Point2], distance: int = 25, degrees_step: int = 5):
    #     points = self.depth_points(center, distance, degrees_step)
    #     distance = [p1.distance_to_point2(p2) for p1, p2 in zip(points, points[1:] + points[:1])]
    #     distance_to_center = [p.distance_to_point2(center) for p in points]
    #     rotate = distance_to_center.index(min(distance_to_center))
    #     points = deque(points)
    #     distance = deque(distance)
    #     distance_to_center = deque(distance_to_center)
    #     points.rotate(-rotate)
    #     distance.rotate(-rotate)
    #     distance_to_center.rotate(-rotate)
    #     points = list(points)
    #     distance_to_center = list(distance_to_center)
    #     points_length = len(points)
    #     chokes = []
    #     _next = 0
    #     deq = deque(points[-int(points_length / 4):], maxlen=int(points_length / 4))
    #     for idx in range(points_length):
    #         if idx < _next:
    #             continue
    #         p = points[idx]
    #         d = distance_to_center[idx]
    #         deq.append(p)
    #         if distance[idx] >= math.radians(degrees_step) * d * 2 + math.sqrt(1.6):
    #             _min = 100
    #             _next = 0
    #             closest_points = (points[idx + 1:] + points)[:int(points_length / 4)]
    #             side_a = p
    #             side_b = points[idx + 1] if idx + 1 < points_length else points[0]
    #             for point, it in zip(closest_points, range(int(points_length / 4))):
    #                 dis = math.hypot(p[0] - point[0], p[1] - point[1])
    #                 if dis < _min:
    #                     _next = it + 1
    #                     _min = dis
    #                     side_b = point
    #             _next += idx
    #             _min = 100
    #             for point in deq:
    #                 dis = math.hypot(side_b[0] - point[0], side_b[1] - point[1])
    #                 if dis < _min:
    #                     _min = dis
    #                     side_a = point
    #             chokes.append([side_a, side_b])
    #     loc_chokes = self.clean_chokes(chokes)
    #     # debug plot
    #     # self.plot(points, loc_chokes, center)
    #     return loc_chokes
    #
    # @staticmethod
    # def plot(points, ch, center):
    #     side_a = set()
    #     side_b = set()
    #     for c in ch:
    #         side_a.add(c[0])
    #         side_b.add(c[1])
    #     plt.scatter(*center, c='#800080', marker='^')
    #     plt.scatter(*zip(*points))
    #     if side_a:
    #         plt.scatter(*zip(*side_a))
    #         plt.scatter(*zip(*side_b))
    #     plt.show()
    #
    # @staticmethod
    # def clean_chokes(chokes: list):
    #     for choke1, choke2 in zip(chokes, chokes[1:] + chokes[:1]):
    #         if choke1[1] == choke2[0]:
    #             choke1[1] = choke2[1]
    #             chokes.remove(choke2)
    #     return chokes
    #
    # def draw_ramp_points(self):
    #     for ramp in self.map_ramps:
    #         height_points = sorted(list({self._bot.get_terrain_z_height(point) for point in ramp.points}))
    #         mid = len(height_points) / 2
    #         if len(height_points) > 2:
    #             top_height = height_points[int(mid + 1)]
    #             bottom_height = height_points[int(mid - 1)]
    #         else:
    #             top_height = height_points[0]
    #             bottom_height = height_points[1]
    #         upper = []
    #         down = []
    #         for p in ramp.points:
    #             h = self._bot.get_terrain_z_height(p)
    #             if h == top_height:
    #                 upper.append(p)
    #                 self._bot.client.debug_box2_out(Point3((*p, h + 0.25)), 0.25, Point3((255, 0, 0)))
    #             elif h == bottom_height:
    #                 down.append(p)
    #                 self._bot.client.debug_box2_out(Point3((*p, h + 0.25)), 0.25, Point3((0, 255, 0)))
    #             else:
    #                 self._bot.client.debug_box2_out(Point3((*p, h + 0.25)), 0.25)
    #         u, d = Point2.center(upper), Point2.center(down)
    #         h1 = self._bot.get_terrain_z_height(u)
    #         h2 = self._bot.get_terrain_z_height(d)
    #         self._bot.client.debug_box2_out(Point3((*u, h1 + 0.25)), 0.25, Point3((0, 0, 255)))
    #         self._bot.client.debug_box2_out(Point3((*d, h2 + 0.25)), 0.25, Point3((0, 0, 255)))
    #
    # def draw_chokes(self):
    #     for ramp in self.map_ramps:
    #         top_center, center, down_center = ramp.centers
    #         vec = (down_center - center).normalized
    #         down = 8 * vec + center
    #         top = -8 * vec + center
    #         h1 = self._bot.get_terrain_z_height(down)
    #         h2 = self._bot.get_terrain_z_height(top)
    #         self._bot.client.debug_sphere_out(Point3((*down, h1)), r=1)
    #         self._bot.client.debug_sphere_out(Point3((*top, h2)), r=1)
    #     # for choke in self.all_chokes:
    #     #     p1, p2 = choke[0], choke[1]
    #     #     h = self._bot.townhalls.first.position3d[2] + 0.1
    #     #     p1 = Point3((*p1, h))
    #     #     p2 = Point3((*p2, h))
    #     #     self._bot.client.debug_line_out(p1, p2)
    #
    # def draw_polygons(self):
    #     siz = self._game_info.placement_grid.data_numpy.shape
    #     placement = np.zeros(siz)
    #     it = 1
    #     for poly in self.polygon_list:
    #         placement[poly.array] = it
    #         it += 1
    #     plt.matshow(placement)
    #     plt.show()

    async def update(self):
        pass

    async def post_update(self):
        pass

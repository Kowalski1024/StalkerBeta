from .manager_base import ManagerBase
from sc2.unit import Unit
from sc2.position import Point2, Point3
from typing import Union, List
import numpy as np
import sc2_math
import math
from collections import deque
from skimage.draw import line_aa
import matplotlib.pyplot as plt


class RegionManager(ManagerBase):
    def __init__(self, bot):
        super().__init__()
        self._bot = bot
        self.all_chokes = self.find_bases_chokes()

    def find_bases_chokes(self):
        chokes = []
        placement = self._bot.game_info.placement_grid.data_numpy.copy()
        plt.matshow(placement)
        plt.show()
        for loc in list(self._bot.expansion_locations_dict.keys()):
            loc_chokes = self.find_chokes(loc)
            for choke in loc_chokes:
                x1, y1 = choke[0][1], choke[0][0]
                x2, y2 = choke[1][1], choke[1][0]
                rr, cc, val = line_aa(x1, y1, x2, y2)
                placement[rr, cc] = 2
        plt.matshow(placement)
        plt.show()
        return chokes

    def depth_points(self, center: Union[tuple, Unit, Point2], dis, degrees_step) -> List[Point2]:
        if isinstance(center, Unit):
            center = center.position
        elif isinstance(center, tuple):
            center = Point2(center)
        placement_grid = self._bot.game_info.pathing_grid.data_numpy
        vec = np.array([[1], [0]])
        point_list = list()
        for deg in range(0, 360, degrees_step):
            x, y = np.matmul(sc2_math.rotation_matrix(deg), vec).round(5)
            x, y = x[0], y[0]
            for m in range(1, dis):
                if not placement_grid[int(x * m + center.y)][int(y * m + center.x)]:
                    point_list.append((int(y * m + center.x), int(x * m + center.y)))
                    break

        # clean list from duplicates
        point_list = list(dict.fromkeys(point_list))
        # clean list from outliers
        distance = [math.hypot(p[0] - center.x, p[1] - center.y) for p in point_list]
        data = sc2_math.get_outliers(np.array(distance))[0]
        data = {i for i in data if distance[i] > np.mean(np.array(distance))}
        return [Point2(i) for j, i in enumerate(point_list) if j not in data]

    def find_chokes(self, center: Union[tuple, Unit, Point2], distance: int = 25, degrees_step: int = 4):
        points = self.depth_points(center, distance, degrees_step)
        distance = [p1.distance_to_point2(p2) for p1, p2 in zip(points, points[1:] + points[:1])]
        distance_to_center = [p.distance_to_point2(center) for p in points]
        rotate = distance_to_center.index(min(distance_to_center))
        points = deque(points)
        distance = deque(distance)
        distance_to_center = deque(distance_to_center)
        points.rotate(-rotate)
        distance.rotate(-rotate)
        distance_to_center.rotate(-rotate)
        points = list(points)
        distance_to_center = list(distance_to_center)
        points_length = len(points)
        chokes = []
        _next = 0
        deq = deque(points[-int(points_length / 4):], maxlen=int(points_length / 4))
        for idx in range(points_length):
            if idx < _next:
                continue
            p = points[idx]
            d = distance_to_center[idx]
            deq.append(p)
            if distance[idx] >= math.radians(degrees_step) * d * 2 + math.sqrt(1.6):
                _min = 100
                _next = 0
                closest_points = (points[idx + 1:] + points)[:int(points_length/4)]
                side_a = p
                side_b = points[idx + 1] if idx+1 < points_length else points[0]
                for point, it in zip(closest_points, range(int(points_length/4))):
                    dis = math.hypot(p[0] - point[0], p[1] - point[1])
                    if dis < _min:
                        _next = it + 1
                        _min = dis
                        side_b = point
                _next += idx
                _min = 100
                for point in deq:
                    dis = math.hypot(side_b[0] - point[0], side_b[1] - point[1])
                    if dis < _min:
                        _min = dis
                        side_a = point
                chokes.append([side_a, side_b])
        loc_chokes = self.clean_chokes(chokes)
        # debug plot
        # self.plot(points, loc_chokes, center)
        return loc_chokes

    @classmethod
    def plot(cls, points, ch, center):
        side_a = set()
        side_b = set()
        for c in ch:
            side_a.add(c[0])
            side_b.add(c[1])
        plt.scatter(*center, c='#800080', marker='^')
        plt.scatter(*zip(*points))
        if side_a:
            plt.scatter(*zip(*side_a))
            plt.scatter(*zip(*side_b))
        plt.show()

    @classmethod
    def clean_chokes(cls, chokes: list):
        for choke1, choke2 in zip(chokes, chokes[1:] + chokes[:1]):
            if choke1[1] == choke2[0]:
                choke1[1] = choke2[1]
                chokes.remove(choke2)
        return chokes

    def draw_chokes(self):
        for choke in self.all_chokes:
            p1, p2 = choke[0], choke[1]
            h = self._bot.townhalls.first.position3d[2] + 0.1
            p1 = Point3((*p1, h))
            p2 = Point3((*p2, h))
            self._bot.client.debug_line_out(p1, p2)

    async def update(self):
        pass

    async def post_update(self):
        pass

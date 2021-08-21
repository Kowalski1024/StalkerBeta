# # 4348182529
# # 4348444673
# # 4348706817
# # 4349231105
# # 4350017537
# # 4350279681
# # 4351066113
# # 4350803969
# # 4350541825
# # 4349755393
# # 4349493249
# # 4348968961
#
#
#
import pickle
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from sc2.position import Point2
from itertools import chain

#
# center = 153.5, 68.5
# points = [(153, 83), (154, 84), (156, 83), (157, 82), (158, 82), (159, 82), (160, 80), (161, 79), (162, 79), (163, 78),
#           (164, 77), (164, 76), (164, 75), (165, 73), (166, 73), (166, 71), (166, 70), (166, 69), (166, 68), (166, 67),
#           (166, 66), (165, 65), (164, 64), (164, 63), (164, 62), (162, 62), (161, 61), (161, 60), (160, 60), (159, 60),
#           (158, 60), (157, 60), (156, 60), (155, 60), (154, 60), (154, 60), (153, 60), (152, 60), (152, 60), (151, 60),
#           (149, 57), (148, 57), (147, 57), (147, 59), (145, 59), (134, 63), (136, 65), (138, 67), (142, 68), (142, 69),
#           (142, 70), (142, 71), (141, 72), (141, 73), (143, 74), (144, 74), (145, 75), (146, 75), (146, 76), (147, 77),
#           (148, 78), (148, 78), (149, 80), (150, 81), (151, 82), (152, 82)]

# center = 91.5, 121.5
# points = [(91, 129), (92, 129), (93, 130), (93, 130), (94, 130), (95, 130), (96, 130), (97, 130), (98, 129), (98, 128),
#           (99, 128), (100, 127), (100, 126), (104, 127), (105, 126), (106, 125), (106, 124), (104, 122), (103, 121),
#           (102, 120), (101, 119), (100, 119), (100, 118), (98, 118), (98, 117), (97, 117), (97, 116), (96, 116),
#           (96, 115), (95, 115), (95, 114), (95, 113), (94, 113), (93, 113), (92, 113), (92, 113), (91, 113), (90, 113),
#           (90, 113), (89, 113), (88, 113), (88, 114), (88, 115), (87, 115), (87, 116), (86, 116), (86, 117), (85, 117),
#           (85, 118), (84, 118), (83, 118), (82, 119), (82, 119), (81, 120), (80, 121), (79, 122), (78, 123), (77, 125),
#           (78, 126), (79, 126), (82, 126), (83, 127), (83, 127), (84, 128), (85, 129), (85, 129), (86, 130), (87, 130),
#           (88, 130), (89, 130), (89, 130), (90, 130)]

center = 57.5, 99.5
points = [(57, 107), (58, 106), (58, 106), (59, 105), (59, 105), (60, 105), (64, 111), (70, 117), (75, 120), (77, 119),
          (78, 117), (79, 114), (80, 112), (81, 110), (70, 104), (68, 102), (66, 101), (65, 100), (65, 99), (66, 98),
          (67, 97), (66, 97), (66, 96), (65, 95), (65, 95), (64, 94), (63, 94), (62, 94), (61, 94), (60, 94), (60, 94),
          (60, 93), (59, 92), (59, 92), (58, 91), (58, 91), (57, 89), (56, 89), (55, 88), (54, 89), (53, 89), (52, 88),
          (42, 72), (37, 71), (34, 71), (32, 74), (30, 77), (29, 79), (29, 83), (44, 93), (46, 95), (46, 96), (45, 97),
          (47, 98), (48, 99), (49, 100), (49, 100), (50, 101), (50, 101), (51, 102), (52, 102), (52, 102), (52, 103),
          (52, 104), (52, 105), (52, 106), (53, 107), (53, 107), (54, 108), (55, 108), (55, 109), (56, 108)]


# def reject_outliers(data, m=1.6) -> np.array:
#     return np.where(abs(data - np.mean(data)) > m * np.std(data))





class Choke:
    def __init__(self, a, b):
        self.side_a = a
        self.side_b = b

    def is_closer(self, other: 'Choke') -> bool:
        d1 = math.hypot(self.side_a[0] - self.side_b[0], self.side_a[1] - self.side_b[1])
        d2 = math.hypot(self.side_a[0] - other.side_b[0], self.side_a[1] - other.side_b[1])
        if d2 < d1:
            return True
        return False

    def __repr__(self):
        return f"({self.side_a}, {self.side_b})"


def draw(points, center):
    points = list(dict.fromkeys(points))
    points_add = points.copy()
    points_add.append(points_add.pop(0))
    distance = [math.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p2, p1 in zip(points, points_add)]
    distance_to_center = [math.hypot(p[0] - center.x, p[1] - center.y) for p in points]
    points = deque(points)
    rotate = distance_to_center.index(min(distance_to_center))
    distance = deque(distance)
    points.rotate(-rotate)
    distance.rotate(-rotate)
    nex = 0
    list_of_chokes = []
    points = list(points)
    points_len = len(points)
    for idx in range(points_len):
        if idx < nex:
            continue
        p = points[idx]
        d = math.hypot(p[0] - center.x, p[1] - center.y)
        if distance[idx] >= math.radians(4) * d * 2 + math.sqrt(1.6):
            print(idx)
            side_a = p
            side_b = points[idx + 1] if idx < points_len else points[0]
            minimum = 100
            it = 0
            nex = 0
            ch = chain(points[idx + 1:], points)
            for point in ch:
                if it == int(points_len/4):
                    break
                it += 1
                dis = math.hypot(p[0] - point[0], p[1] - point[1])
                if dis < minimum:
                    nex = it
                    minimum = dis
                    side_b = point
            nex += idx
            print(nex)
            list_of_chokes.append(Choke(side_a, side_b))

    # list_of_chokes2 = list_of_chokes.copy()
    # list_of_chokes2.append(list_of_chokes2.pop(0))
    # for choke1, choke2 in zip(list_of_chokes.copy(), list_of_chokes2):
    #     print(choke1)
    #     if choke1.is_closer(choke2):
    #         choke1.side_b = choke2.side_b
    #         list_of_chokes.remove(choke2)
    # while points[0] in side_b:
    #     points.rotate(-1)
    side_a = set()
    side_b = set()
    for ch in list_of_chokes:
        side_a.add(ch.side_a)
        side_b.add(ch.side_b)

    plt.scatter(*center, c='#800080', marker='^')
    plt.scatter(*zip(*points))
    plt.scatter(*zip(*side_a))
    plt.scatter(*zip(*side_b))
    plt.show()


center = Point2(center)
draw(points, center)
#
#
# points = list(dict.fromkeys(points))
# points_add = points.copy()
# points_add.append(points_add.pop(0))
# distance = [math.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p2, p1 in zip(points, points_add)]
# points = deque(points)
# rotate = distance.index(min(distance))
# distance = deque(distance)
# points.rotate(-rotate)
# distance.rotate(-rotate)
# dat = list()
# for idx in range(len(points)):
#     p = points[idx]
#     d = math.hypot(p[0] - center[0], p[1] - center[1])
#     if distance[idx] >= math.radians(5)*d + math.sqrt(1.6):
#         dat.append(idx)
#         if idx+1 == len(points):
#             dat.append(0)
#         else:
#             dat.append(idx+1)
#
#
# # points = deque(points)
# # rotate = distance.index(min(distance))
# # distance = deque(distance)
# # points.rotate(-rotate)
# # distance.rotate(-rotate)
# # print(distance)
# # points = list(points)
# # distance = list(distance)
# # deq_list = deque([distance[-1]], maxlen=3)
# # dat = list()
# # for idx in range(len(points)):
# #     deq_list.append(distance[idx])
# #     rej = reject_outliers(np.array(deq_list))[0]
# #     print(rej)
# #     if 2 in rej or distance[idx] > 2:
# #         dat.append(idx)
# #         dat.append(idx+1)
#
#
# #
# # dat = reject_outliers(np.array(distance))[0]
# # dat = [i for i in dat if distance[i] > np.mean(np.array(distance))]
# # print(dat)
# del_points = [points[d] for d in dat]
# points = [i for j, i in enumerate(points) if j not in dat]
#
# plt.scatter(*center, c='#800080', marker='^')
# plt.scatter(*zip(*points))
# if del_points:
#     plt.scatter(*zip(*del_points))
# plt.show()

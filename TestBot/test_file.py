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


def vectors_angle(vector_1: Point2, vector_2: Point2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


p1 = Point2((0, 1))
p2 = Point2((1, 0))
p3 = Point2((0, -1))
x, y = p3
print(vectors_angle(p3, p2))
print(vectors_angle(p1, p2))


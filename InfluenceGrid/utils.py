from typing import Union, List
import numpy as np

from sc2.position import Point2


def circles(radius: List[int], center: Union[tuple, Point2], shape) -> list:
    xx, yy = np.ogrid[:shape[0], :shape[1]]
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    indices = []
    for r in radius:
        indices.append(np.nonzero(circle <= r ** 2))
    return indices


def square(radius, center: Union[tuple, Point2], shape):
    x_start = max(int(center[0] - radius), 0)
    y_start = max(int(center[1] - radius), 0)
    x_arr = []
    y_arr = []
    for x in range(x_start, min(int(center[0] + radius), shape[0])):
        for y in range(y_start, min(int(center[1] + radius), shape[1])):
            x_arr.append(x)
            y_arr.append(y)
    return np.array(x_arr), np.array(y_arr)


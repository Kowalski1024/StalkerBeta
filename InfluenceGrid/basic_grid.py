from typing import List
import numpy as np
from scipy.signal import convolve2d
from sc2.position import Point2
from .utils import circles, square


class BasicGrid:
    def __init__(self, shape, base_val=0, dtype=np.float32, ndarray: np.ndarray = None):
        self.shape = shape
        if ndarray is None:
            self.ndarray = np.full(shape=shape, dtype=dtype, fill_value=base_val)
        else:
            self.ndarray = ndarray.astype(dtype)
            if base_val:
                self.ndarray[self.ndarray == 0] = base_val

    def __add__(self, other):
        ndarray = self.ndarray + other.ndarray
        return BasicGrid(self.shape, ndarray=ndarray)

    def __sub__(self, other):
        ndarray = self.ndarray - other.ndarray
        return BasicGrid(self.shape, ndarray=ndarray)

    def __mul__(self, other):
        ndarray = self.ndarray*other
        return BasicGrid(self.shape, ndarray=ndarray)

    def __truediv__(self, other):
        ndarray = self.ndarray / other
        return BasicGrid(self.shape, ndarray=ndarray)

    def convolve(self, k):
        kernel = np.ones((k, k))
        sum_ = convolve2d(self.ndarray, kernel, mode='same')
        return BasicGrid(self.shape, ndarray=sum_)

    def argmax(self) -> Point2:
        return Point2(np.unravel_index(np.argmax(self.ndarray), self.shape))

    def argsmax(self, k) -> List[Point2]:
        max_ = np.max(self.ndarray)
        return [Point2(p) for p in np.argwhere(self.ndarray == max_)][:k]

    def set_polygon(self, val, poly: np.ndarray):
        if poly.shape == self.shape:
            self.ndarray[poly != 0] = val
        elif reversed(poly.shape) == self.shape:
            self.ndarray[poly.T != 0] = val

    def add_polygon(self, val, poly: np.ndarray):
        if poly.shape == self.shape:
            self.ndarray[poly != 0] += val
        elif reversed(poly.shape) == self.shape:
            self.ndarray[poly.T != 0] += val

    def set_points(self, val, ps):
        for p in ps:
            self.ndarray[p.rounded] = val

    def add_points(self, val, ps):
        for p in ps:
            self.ndarray[p.rounded] += val

    def set_square(self, val, radius, center):
        self.ndarray[square(radius, center, self.shape)] = val

    def add_square(self, val, radius, center):
        self.ndarray[square(radius, center, self.shape)] += val

    def set_circle(self, val, radius, center):
        if not isinstance(val, list):
            val = [val]
            radius = [radius]
        cir = circles(radius, center, self.shape)
        for i, v in enumerate(val):
            self.ndarray[cir[i]] = v

    def add_circle(self, val, radius, center):
        if not isinstance(val, list):
            val = [val]
            radius = [radius]
        cir = circles(radius, center, self.shape)
        for i, v in enumerate(val):
            self.ndarray[cir[i]] += v

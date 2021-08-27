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

from skimage.draw import line


img = np.zeros((10, 10), dtype=np.uint8)
rr, cc = line(1, 1, 8, 8)
print(*zip(rr, cc))


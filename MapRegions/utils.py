import numpy as np

from .destructibles import *
from sc2.unit import Unit


# from MapAnalyzer
def change_destructable_status_in_grid(grid: np.ndarray, unit: Unit, status: int):
    """
    Set destructable positions to status, modifies the grid in place
    """
    type_id = unit.type_id
    pos = unit.position
    name = unit.name

    # this is checked with name because the id of the small mineral destructables
    # has changed over patches and may cause problems
    if name == "MineralField450":
        x = int(pos[0]) - 1
        y = int(pos[1])
        grid[x:(x + 2), y] = status
    elif type_id in destructable_2x2:
        w = 2
        h = 2
        x = int(pos[0] - w/2)
        y = int(pos[1] - h/2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_2x4:
        w = 2
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_2x6:
        w = 2
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_4x2:
        w = 4
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_4x4:
        w = 4
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), (y + 1):(y + h - 1)] = status
        grid[(x + 1):(x + w - 1), y:(y + h)] = status
    elif type_id in destructable_6x2:
        w = 6
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_6x6:
        # for some reason on some maps like death aura
        # these rocks have a bit weird sizes
        # on golden wall this is an exact match
        # on death aura the height is one coordinate off
        # and it varies whether the position is centered too high or too low
        w = 6
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), (y + 1):(y + h - 1)] = status
        grid[(x + 1):(x + w - 1), y:(y + h)] = status
    elif type_id in destructable_12x4:
        w = 12
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_4x12:
        w = 4
        h = 12
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x:(x + w), y:(y + h)] = status
    elif type_id in destructable_BLUR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6):(x_ref + 6 + 2), y_pos + 4] = status
        grid[(x_ref + 5):(x_ref + 5 + 4), y_pos + 3] = status
        grid[(x_ref + 4):(x_ref + 4 + 6), y_pos + 2] = status
        grid[(x_ref + 3):(x_ref + 3 + 7), y_pos + 1] = status
        grid[(x_ref + 2):(x_ref + 2 + 7), y_pos] = status
        grid[(x_ref + 1):(x_ref + 1 + 7), y_pos - 1] = status
        grid[(x_ref + 0):(x_ref + 0 + 7), y_pos - 2] = status
        grid[(x_ref + 0):(x_ref + 0 + 6), y_pos - 3] = status
        grid[(x_ref + 1):(x_ref + 1 + 4), y_pos - 4] = status
        grid[(x_ref + 2):(x_ref + 2 + 2), y_pos - 5] = status

    elif type_id in destructable_ULBR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6):(x_ref + 6 + 2), y_pos - 5] = status
        grid[(x_ref + 5):(x_ref + 5 + 4), y_pos - 4] = status
        grid[(x_ref + 4):(x_ref + 4 + 6), y_pos - 3] = status
        grid[(x_ref + 3):(x_ref + 3 + 7), y_pos - 2] = status
        grid[(x_ref + 2):(x_ref + 2 + 7), y_pos - 1] = status
        grid[(x_ref + 1):(x_ref + 1 + 7), y_pos] = status
        grid[(x_ref + 0):(x_ref + 0 + 7), y_pos + 1] = status
        grid[(x_ref + 0):(x_ref + 0 + 6), y_pos + 2] = status
        grid[(x_ref + 1):(x_ref + 1 + 4), y_pos + 3] = status
        grid[(x_ref + 2):(x_ref + 2 + 2), y_pos + 4] = status
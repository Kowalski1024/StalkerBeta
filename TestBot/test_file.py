
# 4348182529
# 4348444673
# 4348706817
# 4349231105
# 4350017537
# 4350279681
# 4351066113
# 4350803969
# 4350541825
# 4349755393
# 4349493249
# 4348968961
import pickle
from pathlib import Path


def save_obj(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def open_obj(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Fruit:
    value = 30

my_file = Path("sc2_math.py")
if my_file.is_file():
    open_obj('test.zip')
else:
    print('no')

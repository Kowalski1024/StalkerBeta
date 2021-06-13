from region import Region
import matplotlib.pyplot as plt
import sc2_math
import numpy as np

reg = Region(shape=(10, 10), value_type=int)
print(sc2_math.points_in_square_np(1, (2,2), (10,10)))

# print(mat)
# reg.region[np.nonzero(mat)] = 2
# reg.draw_region()


# sc2_math.points_in_square_np(2, (4,4), (10,10))
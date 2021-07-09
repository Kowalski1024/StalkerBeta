from MapInfluence.region import Region

reg = Region(shape=(10, 10), value_type=int)
reg.add_weight_set(5, {(1,2), (3,2)})
reg.draw_region()

# print(mat)
# reg.region[np.nonzero(mat)] = 2
# reg.draw_region()


# sc2_math.points_in_square_np(2, (4,4), (10,10))
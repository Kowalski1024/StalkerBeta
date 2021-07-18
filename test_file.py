from MapInfluence.influence_grid import InfluenceGrid

reg = InfluenceGrid(shape=(10, 10), value_type=int)
reg.add_points_with_weight(5, {(1,2), (3,2)})
reg.show_grid()




# sc2_math.points_in_square_np(2, (4,4), (10,10))
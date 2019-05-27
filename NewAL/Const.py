# A constant file that stores constant parameters
# constant parameter can be use directly in the project


# A map from a custom label to a number label
custom_label = \
      { (1,): 0,
        (1, 2): 1,
        (1, 3): 2,
        (1, 2, 3): 3,
        (2): 4,
        (3): 5 }

user_hypo_table = \
      [
            [3, 4, 5],
            [1, 4, 5],
            [2, 4, 5],
            [0, 4, 5]
      ]
num_hypo = 4
num_feature = 3
num_label = 6

p_teacher_x_h = None
p_teacher_xy_h = None
p_learner_h_xy = None

p_y_xh = \
      [
            [[0.04, 0.2, 0.2, 1], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1  (0), x= 1,2,3
            [[0.16, 0.8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,2  (1), x= 1,2,3
            [[0.16, 0, 0.8, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,3   (2), x= 1,2,3
            [[0.64, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,2,3   (3), x= 1,2,3
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],  # y=2   (4), x= 1,2,3
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]  # y=3   (5), x= 1,2,3
      ]
delta_g_h = None
phx = None

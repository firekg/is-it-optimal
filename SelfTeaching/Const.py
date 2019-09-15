# A constant file that stores constant parameters
# constant parameter can be use directly in the project

'''
# A map from a custom label to a number label
label_map = \
      { "1": 0,
        "1,2": 1,
        "1,3": 2,
        "1,2,3": 3,
        "2": 4,
        "3": 5 }

user_hypo_table = \
      [
            ["1,2,3", "2", "3"],
            ["1,2", "2", "3"],
            ["1,3", "2", "3"],
            ["1", "2", "3"]
      ]

num_hypo = 4
num_feature = 3
num_label = 6

p_teacher_x_h = None
p_teacher_xy_h = None
p_learner_h_xy = None

p_y_xh = \
      [
            # h0,   h1,  h2, h3
            [[0.04, 0.2, 0.2, 1], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1  (0), x= 1,2,3
            [[0.16, 0.8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,2  (1), x= 1,2,3
            [[0.16, 0, 0.8, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,3   (2), x= 1,2,3
            [[0.64, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # y=1,2,3   (3), x= 1,2,3
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],  # y=2   (4), x= 1,2,3
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]  # y=3   (5), x= 1,2,3
      ]
delta_g_h = None
phx = None
'''

# A map from a custom label to a number label
label_map = None

user_hypo_table = None

num_hypo = None
num_feature = None
num_label = None

p_teacher_x_h = None
p_teacher_xy_h = None
p_learner_h_xy = None

p_y_xh = None
delta_g_h = None
phx = None

import numpy
import Const


# Set a user defined hypothesis
# Return:
#        num_hypo: number of hypothesis
#        num_feature: number of features
#        num_label: number of labels
#        p_teacher_x_h: PT(x|h)
#        p_teacher_xy_h: PT(x,y|h)
def Set(user_hypo_map, delta_table=None, knowledgeability=1.0):
      # Get number_hypo, number_feature, num_label automatically
      num_hypo = len(user_hypo_map) if Const.num_hypo is None else Const.num_hypo
      num_feature = len(user_hypo_map[0]) if Const.num_feature is None else Const.num_feature

      if Const.num_label is None:
            list = []
            for hypo in user_hypo_map:
                  for feature in hypo:
                        if feature not in list:
                              list.append(feature)
            num_label = len(list)
      else:
            num_label = Const.num_label

      # if the variable is not defined in const.py
      if Const.p_y_xh is None:
            p_y_xh = numpy.zeros((num_label, num_feature, num_hypo), dtype=float)
            for h in range(num_hypo):
                  for x in range(num_feature):
                        for y in range(num_label):
                              p_y_xh[y, x, h] = 1.0 if user_hypo_map[h][x] == y else 0.0
      else:
            p_y_xh = numpy.array(Const.p_y_xh)

      phx = numpy.zeros((num_hypo), dtype=float)
      for h in range(num_hypo):
            phx[h] = 1 / num_hypo

      # If there is no user delta table (knowledgeability table), then use the default one
      # Else, use the user delta table
      if delta_table is None:
            delta_g_h = numpy.zeros((num_hypo, num_hypo), dtype=float)
            delta_g_h.fill((1 - knowledgeability) / (num_hypo - 1) if num_hypo > 1 else 1)
            numpy.fill_diagonal(delta_g_h, knowledgeability)
      else:
            delta_g_h = delta_table

      p_teacher_x_h = numpy.zeros((num_feature, num_hypo), dtype=float)
      p_teacher_xy_h = numpy.zeros((num_feature, num_label, num_hypo), dtype=float)
      p_learner_h_xy = numpy.zeros((num_hypo, num_feature, num_label))

      #print(delta_table)
      return num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h, phx

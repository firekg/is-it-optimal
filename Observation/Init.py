import numpy as np


# When the equation 5a) and 5b) are applied
# Initial step will be PL(h|xy) = P(y|x,h) PL(h)
def Initstep(number_hypos, number_features, number_labels, p_y_xh):
      p_l_h_xy = np.zeros((number_hypos, number_features, number_labels), dtype=float)
      for x in range(number_features):
            for y in range(number_labels):
                  for h in range(number_hypos):
                        count = 0
                        for hx in range(number_hypos):
                              if (p_y_xh[y, x, h] == p_y_xh[y, x, hx]):
                                    count += 1
                        p_l_h_xy[h, x, y] = p_y_xh[y, x, h] * (1 / count) / number_hypos
      return p_l_h_xy


# Set a user defined hypothesis
# Return:
#        num_hypo: number of hypothesis
#        num_feature: number of features
#        num_label: number of labels
#        p_teacher_x_h: PT(x|h)
#        p_teacher_xy_h: PT(x,y|h)
def Set(user_hypo_map, knowledgeability=1.0):
      # Get number_hypo, number_feature, num_label automatically
      num_hypo = len(user_hypo_map)
      num_feature = len(user_hypo_map[0])
      list = []
      for hypo in user_hypo_map:
            for feature in hypo:
                  if feature not in list:
                        list.append(feature)
      num_label = len(list)

      h_unexplored = num_hypo
      p_y_xh = np.zeros((num_label, num_feature, num_hypo), dtype=int)
      delta_g_h = np.zeros((num_hypo, num_hypo), dtype=float)
      for h in range(num_hypo):
            for x in range(num_feature):
                  for y in range(num_label):
                        p_y_xh[y, x, h] = 1 if user_hypo_map[h][x] == y else 0
      delta_g_h.fill((1 - knowledgeability) / (num_hypo - 1) if num_hypo > 1 else 1)
      np.fill_diagonal(delta_g_h, knowledgeability)
      p_teacher_x_h = np.zeros((num_feature, num_hypo), dtype=float)
      p_teacher_xy_h = np.zeros((num_feature, num_label, num_hypo), dtype=float)
      p_learner_h_xy = np.zeros((num_hypo, num_feature, num_label))
      return num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h

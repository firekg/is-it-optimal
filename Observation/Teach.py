import numpy as np
import Normalize


# eq. 5b)
# PT(x,y | h) = PL(h|x,y) PT(x,y)
# The final table will be a 3D table
def PTeacher_xy_h(number_hypos, number_features, number_labels, table_prob_t_xy_h, table_prob_l_h_xy):
      # The value of PT(x,y)
      prob_t_xy = 1 / number_features / number_labels
      for h in range(number_hypos):
            for x in range(number_features):
                  for y in range(number_labels):
                        table_prob_t_xy_h[x, y, h] = table_prob_l_h_xy[h, x, y] * prob_t_xy
      return


# eq. 6b)
# PT(x|g) = Sum_y  ( PL(g|x,y) * PT(x,y) )
# 'K' stands for the knowledgebility model
def K_PTeacher_xh(number_hypo, number_feature, number_label, p_teacher_xyh, p_teacher_x_h, p_learner_h_xy, delta_gh):
      ptxy = 1 / number_feature / number_label
      temp_p_teacher_xy_h = np.zeros((number_feature, number_label, number_hypo), dtype=float)

      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        p_teacher_xyh[x, y, h] = p_learner_h_xy[h, x, y] * ptxy
      Normalize.Norm_Teacher(number_hypo, number_feature, number_label, p_teacher_xyh)
      temp_teacher_x_h = np.sum(p_teacher_xyh, axis=1)
      for h in range(number_hypo):
            for x in range(number_feature):
                  sum = 0
                  for g in range(number_hypo):
                        sum += temp_teacher_x_h[x, g] * delta_gh[g, h]
                  p_teacher_x_h[x, h] = sum
      Normalize.Norm_Teacher_X(number_hypo, number_feature, number_label, p_teacher_x_h)
      return

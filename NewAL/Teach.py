import numpy


# eq. 6b)
# PT(x|g) = Sum_y  ( PL(g|x,y) * PT(x,y) )
# 'K' stands for the knowledgebility model
def K_PTeacher_xh(number_hypo, number_feature, number_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_gh):
      ptxy = 1 / number_feature / number_label

      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        p_teacher_xy_h[x, y, h] = p_learner_h_xy[h, x, y] * ptxy

      # **** Normalization ****
      Norm_table = numpy.sum(p_teacher_xy_h, axis=(0, 1))
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        if (Norm_table[h] == 0):
                              continue
                        p_teacher_xy_h[x, y, h] /= Norm_table[h]
      # **** Normalization Ends ****

      temp_teacher_x_h = numpy.sum(p_teacher_xy_h, axis=1)
      for h in range(number_hypo):
            for x in range(number_feature):
                  sum = 0
                  for g in range(number_hypo):
                        sum += temp_teacher_x_h[x, g] * delta_gh[g, h]
                  p_teacher_x_h[x, h] = sum

      # **** Normalization ****
      Norm_x = numpy.sum(p_teacher_x_h, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  if (Norm_x[h] == 0):
                        continue
                  p_teacher_x_h[x, h] /= Norm_x[h]
      # **** Normalization Ends ****
      return

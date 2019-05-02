import numpy


# When the equation 5a) and 5b) are applied
# Initial step will be PL(h|xy) = P(y|x,h) PL(h)
def Init_step(number_hypo, number_feature, number_label, p_y_xh, phx):
      p_learner_h_xy = numpy.zeros((number_hypo, number_feature, number_label), dtype=float)
      for x in range(number_feature):
            for y in range(number_label):
                  for h in range(number_hypo):
                        p_learner_h_xy[h, x, y] = p_y_xh[y, x, h] * phx[h] / number_hypo

      # **** Normalization ****
      Norm_h = numpy.sum(p_learner_h_xy, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        if (Norm_h[x, y] == 0):
                              continue
                        p_learner_h_xy[h, x, y] = p_learner_h_xy[h, x, y] / Norm_h[x, y]
      # **** Normalization Ends ****
      return p_learner_h_xy


# Calculate the PL(h|x,y) = P(y|x,h) PL(h) PT(x|h)
# 'K' stands for the knowledgebility model
def K_PLearner_h_xy(number_hypo, number_feature, number_label, p_yxh, p_learner_h_xy, p_teacher_xh, phx):
      prob_learner_h = 1 / number_hypo
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        p_learner_h_xy[h, x, y] = p_yxh[y, x, h] * phx[h] * p_teacher_xh[x, h]

      # **** Normalization ****
      Norm_h = numpy.sum(p_learner_h_xy, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        if (Norm_h[x, y] == 0):
                              continue
                        p_learner_h_xy[h, x, y] = p_learner_h_xy[h, x, y] / Norm_h[x, y]
      # **** Normalization Ends ****

      return

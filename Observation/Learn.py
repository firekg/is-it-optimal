import Normalize

# eq.5a)
# PL(h|x,y) =  PL(h) PT(x,y|h)
def PLearner_h_xy(number_hypos, number_features, number_labels, p_learner_h_xy, p_y_xh, p_teacher_xy_h):
      for x in range(number_features):
            for y in range(number_labels):
                  for h in range(number_hypos):
                        count = 0;
                        for hx in range(number_hypos):
                              if (p_y_xh[y, x, h] == p_y_xh[y, x, hx]):
                                    count = count + 1
                        p_learner_h_xy[h, x, y] = (1 / count) * p_y_xh[y, x, h] * p_teacher_xy_h[x, y, h]
      return


# Calculate the PL(h|x,y) = P(y|x,h) PL(h) PT(x|h)
# 'K' stands for the knowledgebility model
def K_PLearner_h_xy(number_hypo, number_feature, number_label, p_yxh, p_leaner_hxy, p_teacher_xh):
      prob_learner_h = 1 / number_hypo
      for x in range(number_feature):
            for y in range(number_label):
                  for h in range(number_hypo):
                        count = 0;
                        for g in range(number_hypo):
                              if (p_yxh[y, x, h] == p_yxh[y, x, g]):
                                    count = count + 1
                        p_leaner_hxy[h, x, y] = p_yxh[y, x, h] * (1 / count) * p_teacher_xh[x, y, h]
      Normalize.K_Norm_Learner(number_hypo, number_feature, number_label, p_leaner_hxy)
      return

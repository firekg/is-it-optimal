import Normalize


# Calculate the PL(h|x,y) = P(y|x,h) PL(h) PT(x|h)
# 'K' stands for the knowledgebility model
def K_PLearner_h_xy(number_hypo, number_feature, number_label, p_yxh, p_leaner_hxy, p_teacher_xh, phx):
      prob_learner_h = 1 / number_hypo
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        p_leaner_hxy[h, x, y] = p_yxh[y, x, h] * phx[h] * p_teacher_xh[x, h]
      Normalize.K_Norm_Learner(number_hypo, number_feature, number_label, p_leaner_hxy)
      return
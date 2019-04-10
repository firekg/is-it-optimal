import numpy as np


def Norm_Teacher(number_hypo, number_feature, number_labels, p_teacher_xy_h):
      Norm_table = np.sum(p_teacher_xy_h, axis=(0, 1))
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_labels):
                        if (Norm_table[h] == 0):
                              continue
                        p_teacher_xy_h[x, y, h] /= Norm_table[h]
      return


def Norm_Learner(number_hypo, number_feature, number_labels, p_learner_h_xy):
      Normalize_table_xy = np.sum(p_learner_h_xy, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_labels):
                        if (Normalize_table_xy[x, y] == 0):
                              continue
                        p_learner_h_xy[h, x, y] = p_learner_h_xy[h, x, y] / Normalize_table_xy[x, y]
      return


def Norm_Teacher_X(number_hypo, number_feature, number_labels, p_teacher_xh):
      Norm_x = np.sum(p_teacher_xh, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  if (Norm_x[h] == 0):
                        continue
                  p_teacher_xh[x, h] /= Norm_x[h]
      return


def K_Norm_Learner(number_hypo, number_feature, number_labels, p_learner_h_xy):
      Norm_h = np.sum(p_learner_h_xy, axis=0)
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_labels):
                        if (Norm_h[x, y] == 0):
                              continue
                        p_learner_h_xy[h, x, y] = p_learner_h_xy[h, x, y] / Norm_h[x, y]
      return

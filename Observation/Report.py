import numpy as np
import copy



def Report(number_hypo, number_feature, number_label, h_table=None, p_teacher_xy_h=None, p_y_xh=None, p_teacher_x_h=None, p_learner_h_xy=None, knowledge_delta=None, format=False):
      # The new table PT(x|h)
      new_prob_teacher_h_x = np.zeros((number_hypo, number_feature))
      # The new table P(y|x,h)
      new_prob_h_x_y = np.zeros((number_hypo, number_feature, number_label))
      # The new table PT(x,y|h)
      new_prob_teacher_h_x_y = np.zeros((number_hypo, number_feature, number_label))

      # Rearrange all the tables in (hypothesis , features, labels) format
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        if p_teacher_xy_h is not None:
                              new_prob_teacher_h_x_y[h, x, y] = p_teacher_xy_h[x, y, h]
                        if p_y_xh is not None:
                              new_prob_h_x_y[h, x, y] = p_y_xh[y, x, h]
                  if p_teacher_x_h is not None:
                        new_prob_teacher_h_x[h, x] = p_teacher_x_h[x, h]
      if h_table is not None:
            print("---- Hypothesis Table ----")
            print(np.array(h_table))
      if knowledge_delta is not None:
            print("---- Delta Table ----")
            print(knowledge_delta)
      if p_teacher_xy_h is not None:
            print("---- PT(x,y|h) ----")
            print(p_teacher_xy_h)
      if p_y_xh is not None:
            print("---- P(y|x,h) ----")
            print(p_y_xh)
      if p_teacher_x_h is not None:
            print("---- PT(x|h) ----")
            print(p_teacher_x_h)
      if p_learner_h_xy is not None:
            print("---- PL(h|x,y) ----")
            print(p_learner_h_xy)
      return


def Report_to_File(file_path=None, h_table=None, p_teacher_xy_h=None, p_y_xh=None, p_teacher_x_h=None, p_learner_h_xy=None):
      if file_path is None:
            f = open("result.txt", mode="a")
      else:
            f = open(file_path, mode="a")
      if h_table is not None:
            f.write("----- Hypothesis Table -----\n")
            f.write(str(np.array(h_table)))
            f.write("\n")
      if p_teacher_xy_h is not None:
            f.write("----- PT(x,y|h) -----\n")
            f.write(str(p_teacher_xy_h))
            f.write("\n")
      if p_y_xh is not None:
            f.write("----- P(y|x,h) -----\n")
            f.write(str(p_y_xh))
            f.write("\n")
      if p_teacher_x_h is not None:
            f.write("----- PT(x|h) -----\n")
            f.write(str(p_teacher_x_h))
            f.write("\n")
      if p_learner_h_xy is not None:
            f.write("----- PL(h|x,y) -----\n")
            f.write(str(p_learner_h_xy))
            f.write("\n")

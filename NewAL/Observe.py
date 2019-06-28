import copy

import numpy


# Get the current index of the hypothesis for the new table
# Original table: the original hypothesis table
# New table: the new table with reduced hypothesis
# hypo idx: the index of the hypothesis in the original table
def Get_Index(original_table, new_table, hypo_idx):
      for i in range(len(new_table)):
            if new_table[i] == original_table[hypo_idx]:
                  return i


# Get the feature with the highest Pt
# observable_feature_set: the set that contains all select-able features
# hypo: current true hypo
def Get_Feature(observable_feature_set, hypo_idx, p_teacher_xh):
      mx_idx = observable_feature_set[0]
      mx_value = p_teacher_xh[mx_idx, hypo_idx]
      for feature in observable_feature_set:
            if p_teacher_xh[feature, hypo_idx] > mx_value:
                  mx_value = p_teacher_xh[feature, hypo_idx]
                  mx_idx = feature
      return mx_idx


# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature: the target features we want to observe
# return:
#      the probability of finding the true hypothesis
#      the label of the feature
def Observe(hypo_map, true_hypo_idx, target_feature_idx, p_learner_h_xy):
      list = []
      # Get the true label
      true_hypo = hypo_map[true_hypo_idx]
      true_label = true_hypo[target_feature_idx]

      for hypo in hypo_map:
            if hypo[target_feature_idx] != true_label:
                  check = False
            else:
                  list.append(hypo)
      return p_learner_h_xy[true_hypo_idx][target_feature_idx][true_label], true_label

import copy

import numpy


def Get_Index(original_table, new_table, hypo_idx):
      # Get the current index of the p_teacher_xh
      for i in range(len(new_table)):
            if new_table[i] == original_table[hypo_idx]:
                  return i


# Get the feature with the highest Pt
# observable_feature_set: the set that contains all select-able features
# hypo: current true hypo
def Get_Feature(observable_feature_set, hypo_idx, p_teacher_xh):
      mx_value = p_teacher_xh[observable_feature_set[0], hypo_idx]
      mx_idx = observable_feature_set[0]
      for feature in observable_feature_set:
            if p_teacher_xh[feature, hypo_idx] > mx_value:
                  mx_value = p_teacher_xh[feature, hypo_idx]
                  mx_idx = feature
      return mx_idx


# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature: the target features we want to observe
def Observe(hypo_map, true_hypo, target_feature_idx):
      list = []

      # Get the true label
      true_label = true_hypo[target_feature_idx]

      for hypo in hypo_map:
            if hypo[target_feature_idx] != true_label:
                  check = False
            else:
                  list.append(hypo)
      return 1 / len(list) if len(list) >= 1 else 0, list

# print(Get_Target_Feature_Set([0, 1, 2], 3))

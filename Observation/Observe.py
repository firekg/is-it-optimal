import copy

import numpy


def Delete_Repeated_Item(set):
      for i in set:
            new_set = copy.deepcopy(set)
            new_set.remove(i)
            for x in new_set:
                  overlap = True
                  for feature in x:
                        if feature not in i: overlap = False
                  if overlap:
                        set.remove(x)
      return


# all_feature_set: the set of all features
# num_observations: the number of observations, which is also the size of the subset
# check_repeat: delete repeated values (always true)
def Get_Target_Feature_Set(all_feature_set, num_observations, check_repeat=True):
      if num_observations == 1:
            whole_list = []
            for feature in all_feature_set:
                  list = []
                  list.append(feature)
                  whole_list.append(list)
            return whole_list
      else:
            return_list = []
            for feature in all_feature_set:
                  new_set = copy.deepcopy(all_feature_set)
                  new_set.remove(feature)
                  new_list = Get_Target_Feature_Set(new_set, num_observations - 1)
                  for i in new_list:
                        i.append(feature)
                  for x in new_list:
                        return_list.append(x)
            if check_repeat: Delete_Repeated_Item(return_list)
            return return_list


def Get_Probability(p_teacher_x_h, target_hypo, target_feature_set):
      list = []
      for feature in target_feature_set:
            list.append(p_teacher_x_h[feature, target_hypo])
      return numpy.sum(list)


# Get the feature with the highest Pt
# observable_feature_set: the set that contains all select-able features
# hypo: current true hypo
def Get_Feature(observable_feature_set, hypo, p_teacher_xh):
      mx_value = p_teacher_xh[observable_feature_set[0], hypo]
      mx_idx = observable_feature_set[0]
      for feature in observable_feature_set:
            if p_teacher_xh[feature, hypo] > mx_value:
                  mx_value = p_teacher_xh[feature, hypo]
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

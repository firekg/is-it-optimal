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


def Get_Probability_Map(p_teacher_x_h, target_hypo, target_feature_set):
      list = []
      for feature in target_feature_set:
            list.append(p_teacher_x_h[feature, target_hypo])
      return numpy.sum(list)


# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature_set: a set of target features that we want to observe
def Observe(hypo_map, true_hypo, target_feature_set):
      list = []
      label_map = { }

      # Get a list of hypothesis
      for feature in target_feature_set:
            label_map[feature] = true_hypo[feature]

      for hypo in hypo_map:
            check = True
            for feature in target_feature_set:
                  if hypo[feature] != label_map[feature]:
                        check = False
            if check:
                  list.append(hypo)
      return 1 if len(list) == 1 else 0


#print(Get_Target_Feature_Set([0, 1, 2], 3))

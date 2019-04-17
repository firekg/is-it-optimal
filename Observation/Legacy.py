# This file contains the function that no longer used in the project
import Normalize
import numpy
import Observe
import Init
import copy
import Teach
import Learn


def Probability_Task(hypo_map, number_obs, number_hypo, number_feature, number_label, p_teacher_x_h):
      prob_list = []

      feature_list = Observe.Get_Target_Feature_Set([0, 1, 2], number_obs)
      print(feature_list)

      # Assume there is a true hypo
      # Get all posible hypothesis in the hypo map
      for hypo in range(number_hypo):
            F = []
            # Choose a feature to observe
            for feature_set in feature_list:
                  # Get the probability that L will select this feature / these features
                  # prob = Observe.Get_Probability_Map(p_teacher_x_h, hypo, feature_set)

                  # Does the learner find the true hypo ?
                  prob_find = Observe.Observe(hypo_map, hypo, feature_set)

                  F.append(prob_find)
            prob_list.append(F)

      return numpy.array(prob_list)


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
      return 1 / len(list) if len(list) >= 1 else 0


# Sum over x
def Get_Prob_Table(number_obs, p_teacher_x_h, prob_list):
      # Again get the feature list
      feature_list = Observe.Get_Target_Feature_Set([0, 1, 2], number_obs)

      # The new probability map with a lenth = number of hypothesis
      new_prob_list = numpy.zeros(len(prob_list))

      for hypo in range(len(prob_list)):
            sum = 0
            for feature in range(len(feature_list)):
                  prob_select = Observe.Get_Probability(p_teacher_x_h, hypo, feature_list[feature])
                  sum += prob_list[hypo, feature] * prob_select
            new_prob_list[hypo] = sum
      return new_prob_list


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

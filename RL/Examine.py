# This file defines the funtions which are used to test policy
import Select
import Observe
import Update
import Generate
import copy


def Apply_Policy_To_Random_Hypo(hypo_subset, number_features, state_action_label_value_map):
      R = 0
      is_end = False
      next_feature = 0
      true_hypothesis = Generate.Get_Hypo(hypo_subset)
      hypo_remaining_set = hypo_subset
      feature_remaining_set = []
      feature_trajectory = []
      current_feature = -1
      current_label = -1
      for i in range(number_features):
            feature_remaining_set.append(i)
      while True:
            if is_end:
                  break
            else:
                  next_feature = Select.MonteCarlo_Select(feature_remaining_set, current_feature, current_label, state_action_label_value_map)
                  Select.Erase_Feature(feature_remaining_set, next_feature)
                  hypo_remaining_set = Observe.Observe_Subset(true_hypothesis, hypo_remaining_set, next_feature)
                  Observe.Clear_Overlap(feature_remaining_set, hypo_remaining_set)
                  is_end = Observe.Check_End(hypo_remaining_set)
                  feature_trajectory.append(next_feature)
                  current_label = true_hypothesis[next_feature]
                  current_feature = next_feature
      return feature_trajectory


def Apply_Policy_To_All_Hypo(hypo_subset, number_features, state_action_label_value_map):
      result_list = []
      for hypos in hypo_subset:
            R = 0
            is_end = False
            next_feature = 0
            true_hypothesis = copy.deepcopy(hypos)
            hypo_remaining_set = hypo_subset
            feature_remaining_set = []
            current_feature = -1
            current_label = -1
            for i in range(number_features):
                  feature_remaining_set.append(i)
            feature_trajectory = []
            while True:
                  if is_end:
                        result_list.append(feature_trajectory)
                        break
                  else:
                        next_feature = Select.MonteCarlo_Select(feature_remaining_set, current_feature, current_label, state_action_label_value_map)
                        Select.Erase_Feature(feature_remaining_set, next_feature)
                        hypo_remaining_set = Observe.Observe_Subset(true_hypothesis, hypo_remaining_set, next_feature)
                        Observe.Clear_Overlap(feature_remaining_set, hypo_remaining_set)
                        is_end = Observe.Check_End(hypo_remaining_set)
                        feature_trajectory.append(next_feature)
                        current_label = true_hypothesis[next_feature]
                        current_feature = next_feature
      return result_list

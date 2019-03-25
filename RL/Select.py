# This file defines the method of selecting the features
import random
import copy


def MonteCarlo_Select(feature_remaining_set, current_feature, current_label, state_action_label_value_map):
      idx = 0
      mx_value = 0.0
      is_start = False
      sz = len(feature_remaining_set)
      for next_feature in feature_remaining_set:
            temp = (current_feature, next_feature, current_label)
            if temp in state_action_label_value_map:
                  if not is_start:
                        idx = next_feature
                        mx_value = state_action_label_value_map[temp][1]
                        is_start = True
                  else:
                        if state_action_label_value_map[temp][1] > mx_value:
                              idx = next_feature
                              mx_value = state_action_label_value_map[temp][1]

      if not is_start:
            idx = feature_remaining_set[random.randint(0, len(feature_remaining_set) - 1)]
      return idx


def MonteCarlo_Epsilon_Select(feature_remaining_set, current_feature, current_label, state_action_value_map, epsilon=0.1):
      idx = 0
      mx_value = 0.0
      is_start = False
      sz = len(feature_remaining_set)
      non_best_set = []
      for next_feature in feature_remaining_set:
            temp = (current_feature, next_feature, current_label)
            if temp not in state_action_value_map:
                  non_best_set.append(next_feature)
            if temp in state_action_value_map:
                  if not is_start:
                        idx = next_feature
                        mx_value = state_action_value_map[temp][1]
                        is_start = True
                  else:
                        if state_action_value_map[temp][1] > mx_value:
                              non_best_set.append(idx)
                              idx = next_feature
                              mx_value = state_action_value_map[temp][1]
                        else:
                              non_best_set.append(next_feature)

      nsz = len(non_best_set)
      if not is_start:
            return feature_remaining_set[random.randint(0, len(feature_remaining_set) - 1)]
      else:
            is_choose_best = True if (random.randint(1, 100) > epsilon * 100) else False
            if (is_choose_best) or (nsz == 0):
                  return idx
            else:
                  return non_best_set[random.randint(0, nsz - 1)]


def Probability_Select(feature_remaining_set, feature_probability_table):
      return feature_remaining_set[Count(feature_probability_table)]


def Erase_Feature(feature_remaining_set, target_feature):
      feature_remaining_set.remove(target_feature)


def Random_Select(feature_remaining_set):
      return feature_remaining_set[random.randint(0, len(feature_remaining_set) - 1)]


def Count(feature_probability_table):
      idx = 0
      x = random.randint(1, 100)
      for prob in feature_probability_table:
            if x <= prob * 100:
                  return idx
            else:
                  x -= prob * 100
                  idx += 1

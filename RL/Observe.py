# This file defines the way learner pick and observe the feature

def Observe_Subset(true_hypothesis, remaining_hypothesis_set, observe_feature_index):
      label = true_hypothesis[observe_feature_index]
      new_remaining_list = []
      for idx in range(len(remaining_hypothesis_set)):
            if remaining_hypothesis_set[idx][observe_feature_index] == label:
                  new_remaining_list.append(remaining_hypothesis_set[idx])
      return new_remaining_list


def Clear_Overlap(feature_remaining_set, remaining_hypothesis_set):
      sz = len(remaining_hypothesis_set)
      for features in feature_remaining_set:
            is_overlap = True
            for i in range(sz - 1):
                  if (remaining_hypothesis_set[i][features] != remaining_hypothesis_set[i + 1][features]):
                        is_overlap = False
                        break
            if is_overlap:
                  feature_remaining_set.remove(features)


def Check_End(remaining_hypothesis_set):
      return True if len(remaining_hypothesis_set) == 1 else False

# This file contains the function that no longer used in the project
import Normalize
import numpy
import Observe
import Init
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
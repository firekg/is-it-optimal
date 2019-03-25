# This file defines the method for generating hypothesis superset and subset
# The superset contains every possible hypothesis.
# The subset chooses a certain amount of hypothesis from the superset

import random
import copy


def Gen_Superset(number_of_features, number_of_labels):
      if number_of_features == 1:
            lst = []
            for lb in range(number_of_labels):
                  hypo = []
                  hypo.append(lb)
                  lst.append(hypo)
            return lst
      else:
            temp_list = Gen_Superset(number_of_features - 1, number_of_labels)
            return_lst = []
            for hypos in temp_list:
                  for lb in range(number_of_labels):
                        temp_hypo = copy.deepcopy(hypos)
                        temp_hypo.append(lb)
                        return_lst.append(temp_hypo)
            return return_lst


def Gen_Subset(superset, subset_length=7):
      lst = []
      Shuffle(superset);
      for i in range(subset_length):
            lst.append(superset[i])
      return lst


def Get_Hypo(subset, index=-1):
      if index == -1:
            return subset[random.randint(0, len(subset) - 1)]
      else:
            return subset[index]


def Shuffle(superset):
      for idx in range(len(superset)):
            random_location = random.randint(0, len(superset) - 1)
            temp = copy.deepcopy(superset[idx])
            superset[idx] = superset[random_location]
            superset[random_location] = temp


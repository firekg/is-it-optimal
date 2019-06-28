# This file contains the function that no longer used in the project
import Report
import numpy
import Observe
import Set
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


# Print the table to the console or a text file
def ReportTable(self, hypo_table=True, p_teacher_xy_h=False, p_y_xh=False, p_teacher_x_h=True, p_learner_h_xy=True, format=False, textcopy=False):
      Report.Report(self.num_hypo, self.num_feature, self.num_label,
                    self.hypo_table if hypo_table == True else None,
                    self.p_teacher_xy_h if p_teacher_xy_h == True else None,
                    self.p_y_xh if p_y_xh == True else None,
                    self.p_teacher_x_h if p_teacher_x_h == True else None,
                    self.p_learner_h_xy if p_learner_h_xy == True else None, None)
      if textcopy:
            Report.Report_to_File(self.hypo_table if hypo_table == True else None,
                                  self.p_teacher_xy_h if p_teacher_xy_h == True else None,
                                  self.p_y_xh if p_y_xh == True else None,
                                  self.p_teacher_x_h if p_teacher_x_h == True else None,
                                  self.p_learner_h_xy if p_learner_h_xy == True else None)


# eq. 6b)
# PT(x|h) = Sum_y Sum_g ( Pl(g|x,y) * PT(x,y) * delta(g|h) )
# 'K' stands for the knowledgebility model
def K_PTeacher_x_h(number_hypo, number_feature, number_label, p_teacher_xh, p_learner_h_xy, delta_gh):
      p_teacher_xy = 1 / number_feature / number_label
      for h in range(number_hypo):
            for x in range(number_feature):
                  sum = 0
                  for y in range(number_label):
                        for g in range(number_hypo):
                              sum += p_learner_h_xy[g, x, y] * p_teacher_xy * delta_gh[g, h]
                  p_teacher_xh[x, h] = sum
      return


# Hypo list --> Hypo map
# Key: hypothesis
# Value: index
def Transform_to_Map(hypo_list):
      h = { }
      for hypo in range(len(hypo_list)):
            h[tuple(hypo_list[hypo])] = hypo
      return h


# Hypo map --> Hypo list
def Transform_to_List(hypo_map):
      h = []
      for hypo in hypo_map:
            h.append(hypo)
      return h


# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature: the target features we want to observe
def Observe(hypo_map, true_hypo_idx, true_hypo, target_feature_idx, p_learner_h_xy):
      list = []
      # Get the true label
      true_label = true_hypo[target_feature_idx]

      for hypo in hypo_map:
            if hypo[target_feature_idx] != true_label:
                  check = False
            else:
                  list.append(hypo)
      return p_learner_h_xy[true_hypo_idx][target_feature_idx][true_label] if len(p_learner_h_xy) > 1 else 1.0, list


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_table, number_hypo, number_feature, number_label, p_teacher_x_h, knowledgeability, iter=100):
      prob_map = { }
      feature_set = []

      # Assume there is a true hypo = hypo
      # Get all posible hypothesis in the hypo map
      for hypo_idx in range(len(hypo_table)):

            # Get the observable feature set
            for f in range(number_feature):
                  feature_set.append(f)
            obs = 0
            prob = []
            select = []
            # Make a copy of the whole hypo table, and transform to a hypo_map
            hypo_map_copy = copy.deepcopy(hypo_table)
            while True:

                  # Pass the hypo_copy to Set function
                  num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h = Set.Set(hypo_map_copy, knowledgeability=knowledgeability)
                  # Get the PT
                  p_learner_h_xy = Set.Initstep(num_hypo, num_feature, num_label, p_y_xh)
                  Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h, iter)

                  # Choose a feature
                  new_hypo_idx = Observe.Get_Index(hypo_table, hypo_map_copy, hypo_idx)
                  feature = Observe.Get_Feature(feature_set, new_hypo_idx, p_teacher_x_h)
                  obs += 1
                  prob_find, hypo_map_copy = Observe.Observe(hypo_map_copy, new_hypo_idx, hypo_table[hypo_idx], feature, p_learner_h_xy)
                  prob.append(prob_find)
                  select.append(feature)

                  # remove the feature in the feature set
                  feature_set.remove(feature)
                  if len(feature_set) == 0:
                        prob_map[hypo_idx] = (prob, select)
                        break
      return prob_map


def Report(number_hypo, number_feature, number_label, h_table=None, p_teacher_xy_h=None, p_y_xh=None, p_teacher_x_h=None, p_learner_h_xy=None, knowledge_delta=None, format=False):
      # The new table PT(x|h)
      new_prob_teacher_h_x = np.zeros((number_hypo, number_feature))
      # The new table P(y|x,h)
      new_prob_h_x_y = np.zeros((number_hypo, number_feature, number_label))
      # The new table PT(x,y|h)
      new_prob_teacher_h_x_y = np.zeros((number_hypo, number_feature, number_label))

      # Rearrange all the tables in (hypothesis , features, labels) format
      for h in range(number_hypo):
            for x in range(number_feature):
                  for y in range(number_label):
                        if p_teacher_xy_h is not None:
                              new_prob_teacher_h_x_y[h, x, y] = p_teacher_xy_h[x, y, h]
                        if p_y_xh is not None:
                              new_prob_h_x_y[h, x, y] = p_y_xh[y, x, h]
                  if p_teacher_x_h is not None:
                        new_prob_teacher_h_x[h, x] = p_teacher_x_h[x, h]
      if h_table is not None:
            print("---- Hypothesis Table ----")
            print(np.array(h_table))
      if knowledge_delta is not None:
            print("---- Delta Table ----")
            print(knowledge_delta)
      if p_teacher_xy_h is not None:
            print("---- PT(x,y|h) ----")
            print(p_teacher_xy_h)
      if p_y_xh is not None:
            print("---- P(y|x,h) ----")
            print(p_y_xh)
      if p_teacher_x_h is not None:
            print("---- PT(x|h) ----")
            print(p_teacher_x_h)
      if p_learner_h_xy is not None:
            print("---- PL(h|x,y) ----")
            print(p_learner_h_xy)
      return


def Report_to_File(file_path=None, *args):
      if file_path is None:
            f = open("result.txt", mode="a")
      else:
            f = open(file_path, mode="a")
      for tables in args:
            if tables is not None:
                  f.write("----- Table -----\n")
                  f.write(str(np.array(tables)))
                  f.write("\n")

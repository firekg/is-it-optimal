import Normalize
import numpy
import Observe
import Init
import copy
import Teach
import Learn


def Normal_Task(loopstep=0, hypo=None, feature=None, label=None, p_teacher_xy_h=None, p_learner_h_xy=None, p_y_xh=None):

      Normalize.Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_learner_h_xy)
      Normalize.Norm_Teacher(hypo, feature, label, p_teacher_xy_h)
      for x in range(loopstep):
            Learn.PLearner_h_xy(hypo, feature, label, p_learner_h_xy, p_y_xh, p_teacher_xy_h)
            Normalize.Norm_Learner(hypo, feature, label, p_learner_h_xy)
            Teach.PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_learner_h_xy)
            Normalize.Norm_Teacher(hypo, feature, label, p_teacher_xy_h)
      return


# Eq. 6a), 6b)
def Knowledgeability_Task(loopstep=0, hypo=None, feature=None, label=None, p_teacher_xy_h=None, p_teacher_x_h=None, p_learner_h_xy=None, p_y_xh=None, delta_g_h=None):
      Normalize.K_Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.K_PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      for loop in range(loopstep):
            # Calculate learner's table
            Learn.K_PLearner_h_xy(hypo, feature, label, p_y_xh, p_learner_h_xy, p_teacher_xy_h)

            # Calculate teacher's table
            Teach.K_PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      return


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_table, number_hypo, number_feature, number_label, p_teacher_x_h, knowledgeability):
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
            # Make a copy of the whole hypo table, and transform to a hypo_map
            hypo_map_copy = copy.deepcopy(hypo_table)
            while True:

                  # Pass the hypo_copy to Set function
                  num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h = Init.Set(hypo_map_copy, knowledgeability=knowledgeability)
                  # Get the PT
                  p_learner_h_xy = Init.Initstep(num_hypo, num_feature, num_label, p_y_xh)
                  Knowledgeability_Task(500, num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h)

                  # Choose a feature
                  new_hypo_idx = Observe.Get_Index(hypo_table, hypo_map_copy, hypo_idx)
                  feature = Observe.Get_Feature(feature_set, new_hypo_idx, p_teacher_x_h)
                  obs += 1
                  num_remaining_hypothesis, hypo_map_copy = Observe.Observe(hypo_map_copy, hypo_table[hypo_idx], feature)
                  prob_find = 1 / num_remaining_hypothesis
                  prob.append(prob_find)
                  # remove the feature in the feature set
                  feature_set.remove(feature)
                  if len(feature_set) == 0:
                        prob_map[hypo_idx] = prob
                        break
      return prob_map


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

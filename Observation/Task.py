import Normalize
import numpy
import Observe
import Init
import copy
import Teach
import Learn


# Eq. 5a), 5b)
def Normal_Task(hypo, feature, label, p_teacher_xy_h, p_learner_h_xy, p_y_xh, num_iteration=0):
      Normalize.Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_learner_h_xy)
      Normalize.Norm_Teacher(hypo, feature, label, p_teacher_xy_h)
      for x in range(num_iteration):
            Learn.PLearner_h_xy(hypo, feature, label, p_learner_h_xy, p_y_xh, p_teacher_xy_h)
            Normalize.Norm_Learner(hypo, feature, label, p_learner_h_xy)
            Teach.PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_learner_h_xy)
            Normalize.Norm_Teacher(hypo, feature, label, p_teacher_xy_h)
      return


# Eq. 6a), 6b)
def Knowledgeability_Task(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h, num_iteration=0):
      Normalize.K_Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      for loop in range(num_iteration):
            # Calculate learner's table
            Learn.K_PLearner_h_xy(hypo, feature, label, p_y_xh, p_learner_h_xy, p_teacher_x_h)

            # Calculate teacher's table
            Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      return


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_table, number_hypo, number_feature, number_label, p_teacher_x_h, knowledgeability, iter = 100):
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
                  num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h = Init.Set(hypo_map_copy, knowledgeability=knowledgeability)
                  # Get the PT
                  p_learner_h_xy = Init.Initstep(num_hypo, num_feature, num_label, p_y_xh)
                  Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h, iter)

                  # Choose a feature
                  new_hypo_idx = Observe.Get_Index(hypo_table, hypo_map_copy, hypo_idx)
                  feature = Observe.Get_Feature(feature_set, new_hypo_idx, p_teacher_x_h)
                  obs += 1
                  prob_find, hypo_map_copy = Observe.Observe(hypo_map_copy, new_hypo_idx, hypo_table[hypo_idx],  feature, p_learner_h_xy)
                  prob.append(prob_find)
                  select.append(feature)

                  # remove the feature in the feature set
                  feature_set.remove(feature)
                  if len(feature_set) == 0:
                        prob_map[hypo_idx] = (prob,select)
                        break
      return prob_map
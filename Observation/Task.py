import Normalize
import numpy
import Observe
import Init
import copy
import Teach
import Learn


# Eq. 6a), 6b)
def Knowledgeability_Task(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h,
                          p_learner_h_xy, p_y_xh, delta_g_h, phx, num_iteration):
      Normalize.K_Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      for loop in range(num_iteration):
            # Calculate learner's table
            Learn.K_PLearner_h_xy(hypo, feature, label, p_y_xh, p_learner_h_xy, p_teacher_x_h, phx)

            # Calculate teacher's table
            Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      return


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_table, number_hypo, number_feature, number_label, p_teacher_x_h, knowledgeability, iter=100):
      prob_map = { }
      select_map = { }
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

            # Set the environment
            num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h, phx = Init.Set(hypo_table, knowledgeability=knowledgeability)
            while True:

                  # Get the PT
                  p_learner_h_xy = Init.Initstep(num_hypo, num_feature, num_label, p_y_xh, phx)
                  Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h, phx, iter)

                  # Choose a feature
                  feature = Observe.Get_Feature(feature_set, hypo_idx, p_teacher_x_h)
                  obs += 1
                  prob_find, true_label = Observe.Observe(hypo_table, hypo_idx, feature, p_learner_h_xy)
                  prob.append(prob_find)
                  select.append(feature)
                  # Assign the p_learner_h_xy to phx

                  for h in range(number_hypo):
                        phx[h] = p_learner_h_xy[h][feature][true_label]
                  # remove the feature in the feature set,
                  # make the same feature only be observed once
                  feature_set.remove(feature)
                  if len(feature_set) == 0:
                        prob_map[hypo_idx] = prob
                        select_map[hypo_idx] = select
                        break
      return prob_map, select_map


def Average_Hypo(prob_map, number_hypos, number_observations):
      y = []
      for obs in range(number_observations):
            sum = 0
            for hypo_index in prob_map:
                  sum += prob_map[hypo_index][obs]
            y.append(sum / number_hypos)
      return y
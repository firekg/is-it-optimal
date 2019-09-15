import numpy
import Observe
import Set
import copy
import Teach
import Learn


# Eq. 6a), 6b)
# Do the task with convergence check
# If the task converges, the iteration will stop
def Knowledgeability_Task(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_y_xh, delta_g_h, phx, num_iteration):
      p_learner_h_xy = Learn.Init_step(hypo, feature, label, p_y_xh, phx)
      Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      temp_p_teacher_x_h = numpy.copy(p_teacher_x_h)
      loop = 0
      while True:

            # Check the convergence every 10 loops (reduce the time on copy arrays)
            if loop % 10 == 0: temp_p_teacher_x_h = numpy.copy(p_teacher_x_h)

            # Calculate learner's table
            Learn.K_PLearner_h_xy(hypo, feature, label, p_y_xh, p_learner_h_xy, p_teacher_x_h, phx)

            # Calculate teacher's table
            Teach.K_PTeacher_xh(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)

            loop += 1
            if loop % 10 == 0:
                  # If converged, stop the iteration
                  if numpy.array_equal(p_teacher_x_h, temp_p_teacher_x_h): break

            # End the loop by checking the number of iterations
            if loop > num_iteration: break
      return p_learner_h_xy


# hypo_table: The map of the hypothesis
# iter: the number of iterations in the Knowledgeability_Task
# return: a new knowledgeability table that based on the number of observation
def NKnowledgeability_Task(hypo_table, number_hypo, number_feature, number_label, knowledgeability, iter=200):
      feature_set = []

      # New knowledgeability table
      # Axis 1: index of observations
      # Axis 2~3: the delta knowledegeability table
      new_knowledgeability_delta_table = numpy.zeros((number_feature + 1, number_hypo, number_hypo), dtype=float)

      # Assume there is a true hypo = hypo
      # Get all posible hypothesis in the hypo map
      for hypo_idx in range(len(hypo_table)):

            # Get the observable feature set
            for f in range(number_feature):
                  feature_set.append(f)
            obs = 0

            # Set the environment
            num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h, phx = Set.Set(hypo_table, None, knowledgeability=knowledgeability)
            while True:

                  for h in range(number_hypo):
                        new_knowledgeability_delta_table[obs][hypo_idx][h] = phx[h]

                  # Get the PT
                  p_learner_h_xy = Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_y_xh, delta_g_h, phx, iter)

                  # Choose a feature
                  feature = Observe.Get_Feature(feature_set, hypo_idx, p_teacher_x_h)
                  obs += 1
                  prob_find, true_label = Observe.Observe(hypo_table, hypo_idx, feature, p_learner_h_xy)

                  # Assign the p_learner_h_xy to phx
                  for h in range(number_hypo):
                        phx[h] = p_learner_h_xy[h][feature][true_label]

                  # remove the feature in the feature set,
                  # make the same feature only be observed once
                  feature_set.remove(feature)

                  if (len(feature_set) == 0):
                        for h in range(number_hypo):
                              new_knowledgeability_delta_table[obs][hypo_idx][h] = phx[h]
                        break
      return new_knowledgeability_delta_table


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_table, number_hypo, number_feature, number_label, k_table, iter):
      prob_map = { }
      select_map = { }
      const_feature_set = []

      # Append all observable features to the feature set
      for f in range(number_feature):
            const_feature_set.append(f)

      # Assume there is a true hypo = hypo
      # Get all posible hypothesis in the hypo map
      for hypo_idx in range(len(hypo_table)):

            # Make a copy to save time
            feature_set = copy.deepcopy(const_feature_set)

            prob = []
            select = []

            # Set the environment
            # Since we have the knowledgeability table, the knowledgeability argument will be ignored
            num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h, phx = Set.Set(hypo_table, k_table, knowledgeability=1)
            while True:

                  # Get the PT
                  p_learner_h_xy = Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_y_xh, k_table, phx, iter)

                  # Pick the feature with the highest PT
                  feature = Observe.Get_Feature(feature_set, hypo_idx, p_teacher_x_h)

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


def Average_Hypo(prob_map, number_hypos):
      y = []
      for obs in range(len(prob_map[0])):
            sum = 0
            for hypo_index in prob_map:
                  sum += prob_map[hypo_index][obs]
            y.append(sum / number_hypos)
      return y

# This file is used for sampling data points ( [K] , p )
# The data points are used for the polynomial regression
# [K] is the knowledgeability matrix

import numpy
import random
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


# Get a random K matrix
# K an n*n matrix where n = number of hypothesis
# Each row should add up to 1.
# Each value in the matrix should be greater or equal than 0 and less or equal than 1
# We generate values with 0.1 accuracy
def Random_K(number_hypo, set_size):
      K_set = []
      for i in range(set_size):
            # Create an n*n matrix with all values 0
            K = numpy.zeros((number_hypo, number_hypo), dtype=float)

            for line in range(number_hypo):
                  upper = 10
                  for row in range(number_hypo):
                        # Generate a value between 0 and upper
                        k = random.randint(0, upper)
                        upper -= k
                        k = k / 10
                        K[line, row] = k
                        if upper == 0: break

                  # Randomly shuffle the values in a row
                  for row in range(number_hypo):
                        i = random.randint(0, number_hypo - 1)
                        # Swap values
                        temp = K[line, row]
                        K[line, row] = K[line, i]
                        K[line, i] = temp
            K_set.append(K)
      return K_set


# The values on the diagonal are the same
# [a,x,x,x]
# [x,a,x,x]
# [x,x,a,x]
# [x,x,x,a]
def Pattern_Diagonal_K(number_hypo, step_size):
      K_set = []
      a = 0
      while True:
            # Create an n*n matrix with all values 0
            K = numpy.zeros((number_hypo, number_hypo), dtype=float)
            for i in range(number_hypo):
                  K[i, 0:number_hypo] = round((1 - a) / (number_hypo - 1), 5)
                  K[i, i] = a
            K_set.append(K)
            a = round(a + step_size, 5)
            if a > 1: break
      return K_set


# Values form a square shape
# [a,a,0,0]
# [a,a,0,0]
# [0,0,a,a]
# [0,0,a,a]
def Pattern_Cube_K(number_hypo, cube_size):
      K_set = []
      K = numpy.zeros((number_hypo, number_hypo), dtype=float)
      s = number_hypo
      a = 0
      while True:
            # If there is enough place for the next cube then use the cube size
            # else use the remaing space size
            if s >= cube_size:
                  p = round(1 / cube_size, 5)
                  K[a:a + cube_size, a:a + cube_size] = p
            elif s < cube_size:
                  p = round(1 / s, 5)
                  K[a:a + s, a:a + s] = p
            s -= cube_size
            a += cube_size
            if s <= 0: break
      K_set.append(K)
      return K_set


# hypo_map: The map of the hypothesis
# return: p for each pre_determined h
#              label list
def Sample_P(hypo_table, number_hypo, number_feature, number_label, k_matrix, iter):
      hypo_table_size = len(hypo_table)
      prob_list = []
      const_feature_set = []

      # Append all observable features to the feature set
      for f in range(number_feature):
            const_feature_set.append(f)

      # Assume there is a true hypo = hypo
      # Get all posible hypothesis in the hypo map
      for hypo_idx in range(hypo_table_size):
            # Make a copy to save time
            feature_set = copy.deepcopy(const_feature_set)

            # Set the environment
            # Since we have the knowledgeability table, the knowledgeability argument will be ignored
            num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h, phx = Set.Set(hypo_table, k_matrix, knowledgeability=1)

            # Get the PT
            p_learner_h_xy = Knowledgeability_Task(num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_y_xh, k_matrix, phx, iter)

            # Pick the feature with the highest PT
            feature = Observe.Get_Feature(feature_set, hypo_idx, p_teacher_x_h)

            # Get the p and the corresponding label
            prob_find, true_label = Observe.Observe(hypo_table, hypo_idx, feature, p_learner_h_xy)

            # Append the p to the prob list
            prob_list.append(prob_find)

      p = 0
      # Average the value
      for item in prob_list:
            p += item

      p /= (number_hypo)
      return p


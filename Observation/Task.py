import Normalize
import numpy
import Observe
import Init
import copy
import Teach
import Learn


def Normal_Task(loopstep=0, hypo=None, feature=None, label=None, p_teacher_xy_h=None, p_learner_h_xy=None, p_y_xh=None):
      p_learner_h_xy = Init.Initstep(hypo, feature, label, p_y_xh)
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
      p_learner_h_xy = Init.Initstep(hypo, feature, label, p_y_xh)
      Normalize.K_Norm_Learner(hypo, feature, label, p_learner_h_xy)
      Teach.K_PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      for ls in range(loopstep):
            # Calculate learner's table
            Learn.K_PLearner_h_xy(hypo, feature, label, p_y_xh, p_learner_h_xy, p_teacher_xy_h)

            # Calculate teacher's table
            Teach.K_PTeacher_xy_h(hypo, feature, label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, delta_g_h)
      return


# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability
def Probability_Task(hypo_map, number_hypo, number_feature, number_label, p_teacher_x_h):
      prob_map = { }
      feature_set = []

      # Assume there is a true hypo = hypo
      # Get all posible hypothesis in the hypo map
      for hypo_idx in range(len(hypo_map)):

            # Get the observable feature set
            for f in range(number_feature):
                  feature_set.append(f)
            obs = 0
            prob = []
            # Make a copy of the whole hypo map
            hypo_map_copy = copy.deepcopy(hypo_map)
            while True:
                  '''
                  # Pass the hypo_copy to Set function
                  num_hypo, num_feature, num_label, p_teacher_x_h, p_teacher_xy_h, p_learner_h_xy, p_y_xh, delta_g_h = Init.Set(hypo_map_copy)
                  # Get the PT
                  Knowledgeability_Task(1000, num_hypo, num_feature, num_label, p_teacher_xy_h, p_teacher_x_h, p_learner_h_xy, p_y_xh, delta_g_h)
                  '''
                  # Choose a feature
                  feature = Observe.Get_Feature(feature_set, hypo_idx, p_teacher_x_h)
                  obs += 1
                  prob_find, hypo_map_copy = Observe.Observe(hypo_map_copy, hypo_map[hypo_idx], feature)
                  prob.append(prob_find)
                  # remove the feature in the feature set
                  feature_set.remove(feature)
                  if len(feature_set) == 0:
                        prob_map[hypo_idx] = prob
                        break
      return prob_map

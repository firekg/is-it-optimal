import Normalize
import Observe
import Init
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


def Probability_Task(hypo_map, number_obs, number_hypo, number_feature, number_label, p_teacher_x_h):
      list = []

      feature_list = Observe.Get_Target_Feature_Set([0, 1, 2], number_obs)
      print(feature_list)

      # Assume there is a true hypo
      # Get all posible hypothesis in the hypo map
      for hypo in range(number_hypo):
            F = []
            # Choose a feature to observe
            for feature_set in feature_list:
                  # Get the probability that L will select this feature / these features
                  prob = Observe.Get_Probability_Map(p_teacher_x_h, hypo, feature_set)

                  # Does the learner find the true hypo ?
                  prob_t = Observe.Observe(hypo_map, hypo_map[hypo], feature_set)

                  F.append(prob * prob_t)
            list.append(F)

      return list

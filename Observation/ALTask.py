import numpy as np
import Normalize
import Init
import Observe
import Task
import Report
import copy


class ActiveLearning:

      # Initialize all variables that used in the learning task
      # If user_hypo is empty, the program will automatically generate a full boundary task
      def __init__(self, Hyposize=4, Featuresize=4, Labelsize=2, knowledgeability=1.0):
            # Teacher's Knowledgeability
            self.knowledge = knowledgeability

            # Number of total hypothesis (h) features (x) and labels (y)
            self.num_hypo = Hyposize
            self.num_feature = Featuresize
            self.num_label = Labelsize

            # The amount of unexplored Hypothesis
            self.h_unexplored = self.num_hypo

            # The P(y|x,h) table for the boundary task
            self.p_y_xh = np.zeros((self.num_label, self.num_feature, self.num_hypo), dtype=int)

            # Create Teacher's Knowledge table Delta(g|h) based on the knowledgeability
            self.delta_g_h = np.zeros((self.num_hypo, self.num_hypo), dtype=float)

            # Calculate the P(y|x,h)
            for i in range(self.num_feature):
                  self.p_y_xh[1][i][0:self.num_feature - i] = 1
                  self.p_y_xh[0][i][self.num_feature - i:self.num_feature + 1] = 1

            # Create the knowledgeability table
            self.delta_g_h.fill((1 - self.knowledge) / (self.num_hypo - 1))
            np.fill_diagonal(self.delta_g_h, self.knowledge)

            # The empty PT(x|h) table for the boundary task
            self.p_teacher_x_h = np.zeros((self.num_feature, self.num_hypo), dtype=float)

            # The empty PT(x,y|h) table for the boundary task
            self.p_teacher_xy_h = np.zeros((self.num_feature, self.num_label, self.num_hypo), dtype=float)

            # The empty table  PL(h|x,y) for the boundary task
            self.p_learner_h_xy = np.zeros((self.num_hypo, self.num_feature, self.num_label))

            # The hypothesis and feature table
            self.hypo_table = np.zeros((self.num_hypo + 1, self.num_feature), dtype=int)
            for h in range(self.num_hypo + 1): self.hypo_table[h, 0:self.num_hypo - h] = 1

      # Set the user hypothesis and the corresponding environment
      def Set(self, user_hypo, k):
            self.num_hypo, self.num_feature, self.num_label, self.p_teacher_x_h, \
            self.p_teacher_xy_h, self.p_learner_h_xy, self.p_y_xh, self.delta_g_h = Init.Set(user_hypo, k)

      def N_Task(self, loopstep=0):
            Task.Normal_Task(loopstep, self.num_hypo, self.num_feature, self.num_label, self.p_teacher_xy_h, self.p_learner_h_xy, self.p_y_xh)
            return

      def K_Task(self, loopstep=0):
            Task.Knowledgeability_Task(loopstep, self.num_hypo, self.num_feature, self.num_label, self.p_teacher_xy_h, self.p_teacher_x_h, self.p_learner_h_xy, self.p_y_xh, self.delta_g_h)
            return

      def P_Task(self):
            print(self.p_teacher_x_h)
            p_h_x = Task.Probability_Task(self.hypo_table, self.num_hypo, self.num_feature, self.num_label, self.p_teacher_x_h)
            print(p_h_x)
            return

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

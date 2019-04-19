import numpy
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

            # Number of
            #         total hypothesis (h)
            #         features (x)
            #         labels (y)
            self.num_hypo = Hyposize
            self.num_feature = Featuresize
            self.num_label = Labelsize

            # The P(y|x,h) table for the boundary task
            self.p_y_xh = numpy.zeros((self.num_label, self.num_feature, self.num_hypo), dtype=int)

            # Create Teacher's Knowledge table Delta(g|h) based on the knowledgeability
            self.delta_g_h = numpy.zeros((self.num_hypo, self.num_hypo), dtype=float)

            # Calculate the P(y|x,h)
            for i in range(self.num_feature):
                  self.p_y_xh[1][i][0:self.num_feature - i] = 1
                  self.p_y_xh[0][i][self.num_feature - i:self.num_feature + 1] = 1

            # Create the knowledgeability table
            self.delta_g_h.fill((1 - self.knowledge) / (self.num_hypo - 1))
            numpy.fill_diagonal(self.delta_g_h, self.knowledge)

            # The empty PT(x|h) table for the boundary task
            self.p_teacher_x_h = numpy.zeros((self.num_feature, self.num_hypo), dtype=float)

            # The empty PT(x,y|h) table for the boundary task
            self.p_teacher_xy_h = numpy.zeros((self.num_feature, self.num_label, self.num_hypo), dtype=float)

            # The empty table  PL(h|x,y) for the boundary task
            self.p_learner_h_xy = numpy.zeros((self.num_hypo, self.num_feature, self.num_label))

            # The hypothesis and feature table
            self.hypo_table = numpy.zeros((self.num_hypo + 1, self.num_feature), dtype=int)
            for h in range(self.num_hypo + 1): self.hypo_table[h, 0:self.num_hypo - h] = 1

            # The empty p_h_x table
            self.phx = numpy.zeros((self.num_feature), dtype=float)

      # set the user hypothesis and to the current hypothesis table
      # automatically set the corresponding environment (e.g. variables, tables)
      def Set(self, user_hypo):
            self.hypo_table = user_hypo
            self.num_hypo, self.num_feature, self.num_label, self.p_teacher_x_h, self.p_teacher_xy_h, self.p_learner_h_xy, self.p_y_xh, self.delta_g_h, self.phx = Init.Set(user_hypo, self.knowledge)

      # P(get h) based on the number of observations
      def P_Task(self):
            p, s ,sp= Task.Probability_Task(self.hypo_table, self.num_hypo, self.num_feature, self.num_label, self.p_teacher_x_h, self.knowledge)
            print(p, s, sp, sep="\n")
            return

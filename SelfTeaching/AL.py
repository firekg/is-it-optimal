import numpy
import Set
import Sample
import Observe
import Task
import copy


class ActiveLearning:
    '''Initialize all variables that used in the learning task
    If user_hypo is empty, the program will automatically generate a full boundary task'''
    def __init__(self):
        self.knowledge = None   # Teacher's Knowledgeability

        '''Number of
                total hypothesis (h)
                features (x)
                labels (y)'''
        self.num_hypo = None
        self.num_feature = None
        self.num_label = None

        self.p_y_xh = None  # P(y|x,h) matrix
        self.delta_g_h = None  # Teacher's Knowledgeability matrix Delta(g|h)
        self.p_teacher_x_h = None  # PT(x|h) matrix
        self.p_teacher_xy_h = None  # PT(x,y|h) matrix
        self.p_learner_h_xy = None  # PL(h|x,y) matrix
        self.hypo_table = None  # Hypothesis matrix
        self.phx = None  # P_h_x vector

    '''set the user hypothesis and to the current hypothesis table
    automatically set the corresponding environment (e.g. variables, tables)'''
    def Set(self, user_hypo, knowledgeability):
        self.knowledge = knowledgeability
        self.hypo_table = user_hypo
        self.num_hypo, self.num_feature, self.num_label, self.p_teacher_x_h, self.p_teacher_xy_h, self.p_learner_h_xy, self.p_y_xh, self.delta_g_h, self.phx = Set.Set(self.hypo_table, None,
                                                                                                                                                                       self.knowledge)
    '''P(h) based on the number of observations'''
    def P_Task(self, k=None):
        k = self.delta_g_h if k is None else k
        p, s = Task.Probability_Task(
            self.hypo_table, self.num_hypo, self.num_feature, self.num_label, k, 100)
        print(p, s, sep="\n")
        return s
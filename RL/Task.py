import Generate
import Select
import Report
import Observe
import Update
import Examine
import copy


class Subset_Task:
      def __init__(self, number_features=4, number_labels=2):
            self.num_feature = number_features
            self.num_label = number_labels
            self.hypo_superset = Generate.Gen_Superset(number_features, number_labels)
            self.hypo_subset = []
            self.hypo_remaining_set = []
            self.feature_remaining = []
            self.true_hypothesis = []
            self.prob_map = { }
            self.state_action_label_value_map = { }

      def Init_Subset(self, length=1, user_subset=[], show_superset=False, show_subset=True):
            if len(user_subset) == 0:
                  self.hypo_subset = Generate.Gen_Subset(self.hypo_superset, length)
            else:
                  self.hypo_subset = user_subset
            if show_superset:
                  Report.Print_Set(self.hypo_superset)
                  print("----Superset----")
            if show_subset:
                  Report.Print_Set(self.hypo_subset)
                  print("----Subset----")

      def Reset(self):
            self.true_hypothesis = Generate.Get_Hypo(self.hypo_subset)
            self.feature_remaining = []
            self.feature_trajectory = []
            self.state_list = []
            self.hypo_remaining_set = copy.deepcopy(self.hypo_subset)
            for f in range(self.num_feature):
                  self.feature_remaining.append(f)

      def Simulate(self, simulation_loop=1, state_value_report=True):
            for looptime in range(simulation_loop):
                  R = 0
                  is_end = False
                  next_feature = False
                  current_feature = -1
                  current_label = -1
                  self.Reset()
                  while True:
                        if is_end:
                              Update.MonteCarlo_Update(R, self.state_list, self.state_action_label_value_map)
                              break
                        else:
                              next_feature = Select.MonteCarlo_Epsilon_Select(self.feature_remaining, current_feature, current_label, self.state_action_label_value_map)
                              Select.Erase_Feature(self.feature_remaining, next_feature)
                              self.hypo_remaining_set = Observe.Observe_Subset(self.true_hypothesis, self.hypo_remaining_set, next_feature)
                              Observe.Clear_Overlap(self.feature_remaining, self.hypo_remaining_set)
                              is_end = Observe.Check_End(self.hypo_remaining_set)
                              self.state_list.append((current_feature, next_feature, current_label))
                              R += -1
                              current_label = self.true_hypothesis[next_feature]
                              current_feature = next_feature
            if state_value_report:
                  Report.Report_State_Value_Map(self.state_action_label_value_map)

      def Apply(self, apply_loop=1, report_trajectory=True, report_average=False):
            avg = 0
            for loops in range(apply_loop):
                  trajectory = Examine.Apply_Policy_To_Random_Hypo(self.hypo_subset, self.num_feature, self.state_action_label_value_map)
                  if report_average: avg += len(trajectory)
                  if report_trajectory: print(trajectory)
            if report_average: print("The average step is:", avg / apply_loop)

      def Probability_Map(self, probability_map_report=True, probability_map_plot=True, cumulative=True):
            self.prob_map = { }
            for i in range(self.num_feature + 1): self.prob_map[i] = 0
            trajectory_list = Examine.Apply_Policy_To_All_Hypo(self.hypo_subset, self.num_feature, self.state_action_label_value_map)
            for i in trajectory_list:
                  self.prob_map[len(i)] += 1 / len(self.hypo_subset)
            if cumulative:
                  for i in range(self.num_feature): self.prob_map[i + 1] += self.prob_map[i]
            if probability_map_report: Report.Report_Prob_Table(self.prob_map)
            if probability_map_plot: Report.Plot_Prob_Table(self.prob_map)

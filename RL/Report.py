# This file defines some methods for printing out the result
import matplotlib.pyplot as mplot


def Report_State_Value_Map(state_value_map):
      for states in state_value_map:
            print(states, " : ", end=",")
            print("Step: ", state_value_map[states][0], "   |   Average Value: ", state_value_map[states][1], end="")
            print()


def Report_Prob_Table(probability_table, decimal=-1):
      for obs in probability_table:
            print("Observation", obs, " : ", end="")
            if decimal == -1:
                  print("Probability: ", probability_table[obs], end="")
            else:
                  print("Probability: ", round(probability_table[obs], decimal), end="")
            print()


def Plot_Prob_Table(probability_table):
      x = []
      y = []
      for obs in probability_table:
            x.append(obs)
            y.append(probability_table[obs])
      mplot.plot(x, y)
      mplot.show()


def Print_Set(any_set):
      for hypo in any_set:
            for lb in hypo:
                  print(lb, end=",")
            print()

import numpy
import copy

# *****************************************************
# All following hypothesis table has a pattern
# and we uses number of features = 4 as an example
# if you want to see what hypothesis table looks like,
# set printout to be true
# *****************************************************

# 1,1,1,1
# 1,1,1,0
# 1,1,0,0
# 1,0,0,0
# 0,0,0,0
def Boundary_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features):
            table[i, 0:(number_features - i)] = 1
      if printout:
            print(table)
      return table


# 0,0,0,0
# 0,0,0,0
# 0,0,0,0
# 0,0,0,0
def Uniform_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      if printout:
            print(table)
      return table


# 1,0,0,0
# 0,1,0,0
# 0,0,1,0
# 0,0,0,1
def Identity_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 1
      if printout:
            print(table)
      return table


# 0,1,1,1
# 1,0,1,1
# 1,1,0,1
# 1,1,1,0
def Reversed_Indentity_Hypo_Table(number_features, printout=False):
      table = numpy.ones((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 0
      if printout:
            print(table)
      return table


# 1,0,0,0
# 1,1,0,0
# 0,1,1,0
# 0,0,1,1
# 0,0,0,1
def Zigzag_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features + 1):
            table[i][i - 1 if i - 1 >= 0 else 0: i + 1 if i +
                                                          1 <= number_features else number_features] = 1
      if printout:
            print(table)
      return table


# 1,1,0,0,0
# 0,1,1,0,0
# 0,0,1,1,0
# 0,0,0,1,1
def ZigzagTranpose_Hypo_Table(number_features, printout=False):
      h = number_features - 1
      f = number_features
      table = numpy.zeros((h, f), dtype=int)
      for i in range(h):
            table[i][i:i + 2] = 1
      if printout:
            print(table)
      return table


def Transfer_User_Table(user_hypothesis_table, coding_map):
      for a in range(len(user_hypothesis_table)):
            sz = len(user_hypothesis_table[a])
            for i in range(sz):
                  user_hypothesis_table[a][i] = coding_map[user_hypothesis_table[a][i]]

def Purly_Random_Hypo(num_features, num_hypo):
    L = [[0],[1]]
    for x in range(num_features-1):
        K = []
        for elem in L:
            for label in range(2):
                a = copy.deepcopy(elem)
                a.append(label)
                K.append(a)
        L = copy.deepcopy(K)
    for x in range(len(L)):
        p = numpy.random.randint(0, len(L))
        L[x], L[p] = L[p], L[x]
    return numpy.array(L[:num_hypo])
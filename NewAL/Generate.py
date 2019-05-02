import numpy


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
def Generate_Boundary_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features):
            table[i, 0:(number_features - i)] = 1
      if printout: print(table)
      return list(table)


# 0,0,0,0
# 0,0,0,0
# 0,0,0,0
# 0,0,0,0
def Generate_Uniform_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      if printout: print(table)
      return list(table)


# 1,0,0,0
# 0,1,0,0
# 0,0,1,0
# 0,0,0,1
def Generate_Indentity_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 1
      if printout: print(table)
      return list(table)


# 0,1,1,1
# 1,0,1,1
# 1,1,0,1
# 1,1,1,0
def Generate_Reversed_Indentity_Hypo_Table(number_features, printout=False):
      table = numpy.ones((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 0
      if printout: print(table)
      return list(table)


# 1,0,0,0
# 1,1,0,0
# 0,1,1,0
# 0,0,1,1
# 0,0,0,1
def Generate_Zigzag_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features + 1):
            table[i][i - 1 if i - 1 >= 0 else 0: i + 1 if i + 1 <= number_features else number_features] = 1
      if printout: print(table)
      return list(table)

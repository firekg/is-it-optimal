import numpy


def Generate_Boundary_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features):
            table[i, 0:(number_features - i)] = 1
      if printout: print(list(table))
      return list(table)


def Generate_Uniform_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      if printout: print(list(table))
      return list(table)


def Generate_Indentity_Hypo_Table(number_features, printout=False):
      table = numpy.zeros((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 1
      if printout: print(list(table))
      return list(table)

def Generate_Reversed_Indentity_Hypo_Table(number_features, printout=False):
      table = numpy.ones((number_features, number_features), dtype=int)
      for i in range(number_features):
            table[i, i] = 0
      if printout: print(list(table))
      return list(table)
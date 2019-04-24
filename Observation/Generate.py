import numpy


def Generate_Boundary_Hypo_Table(number_features):
      table = numpy.zeros((number_features + 1, number_features), dtype=int)
      for i in range(number_features):
            table[i, 0:(number_features - i)] = 1
      return list(table)

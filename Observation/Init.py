import numpy as np


# When the equation 5a) and 5b) are applied
# Initial step will be PL(h|xy) = P(y|x,h) PL(h)
def Initstep(number_hypos, number_features, number_labels, p_y_xh):
      p_l_h_xy = np.zeros((number_hypos, number_features, number_labels), dtype=float)
      for x in range(number_features):
            for y in range(number_labels):
                  for h in range(number_hypos):
                        count = 0
                        for hx in range(number_hypos):
                              if (p_y_xh[y, x, h] == p_y_xh[y, x, hx]):
                                    count += 1
                        p_l_h_xy[h, x, y] = p_y_xh[y, x, h] * (1 / count) / number_hypos
      return p_l_h_xy

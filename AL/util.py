import numpy as np


def normalize(vec):
    sum_val = np.sum(vec)
    if np.isclose(sum_val, 0):
        # print("Warning: vec sums to zero")
        return vec
    else:
        return vec/sum_val


def entropy(vec):
    sum_val = np.sum(vec)
     # if np.isclose(sum_val, 0):
     #     print("Warning: vec sums to zero")
     # if not np.isclose(sum_val, 1):
     #     print("Warning: vec does not sum up to 1 (or 0).")
    return -np.nansum(vec*np.log(vec))


def performance_curve_difference(curve1, curve2):
    if len(curve1)!=len(curve2):
        raise ValueError("The two input vectors should have same length.")
    n = len(curve1)
    judge = 0
    for i in range(n):
        if np.isclose(curve1[i], curve2[i]):
            judge += 0
        elif curve1[i] > curve2[i]:
            judge += 1
        elif curve1[i] < curve2[i]:
            judge -= 1
    # judge = 0 means a tie,
    # +ive means curve 1 bigger
    # -ive means curve 2 bigger
    return judge

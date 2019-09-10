import copy

def Permutation(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

    
# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature_set: a set of target features that we want to observe
def Observe(hypo_map, true_hypo, target_feature_set):
      list = []
      label_map = { }

      # Get a list of hypothesis
      for feature in target_feature_set:
            label_map[feature] = true_hypo[feature]

      for hypo in hypo_map:
            check = True
            for feature in target_feature_set:
                  if hypo[feature] != label_map[feature]:
                        check = False
            if check:
                  list.append(hypo)
      return 1 / len(list) if len(list) >= 1 else 0

def Radix(numpy_array, start, end, comp):

    return  

def DummySort(numpy_array, start, end, comp):
    for k in range(start, end):
        for p in range(k, end):
            if numpy_array[k, comp] > numpy_array[p, comp]:
                temp = numpy_array[k,:]
                numpy_array[k,:] = numpy_array[p,:]
                numpy_array[p,:] = temp
    return
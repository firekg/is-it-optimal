import copy

# Generating permutation using Heap Algorithm
def Permutation(a, size, n, P):
    # if size becomes 1 then prints the obtained
    # permutation
    if (size == 1):
        P.append(copy.deepcopy(a))
        return

    for i in range(size):
        Permutation(a, size-1, n, P)

        # if size is odd, swap first and last
        # element
        # else If size is even, swap ith and last element
        if size & 1:
            a[0], a[size-1] = a[size-1], a[0]
        else:
            a[i], a[size-1] = a[size-1], a[i]


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
# Tree Model
import numpy
import Utility
import copy
import Generate


# Generate the hypothesis matrix
total_features = 4
hypo = Generate.Boundary_Hypo_Table(total_features, True)
total_hypos = len(hypo)

# Create the observation list
arr = numpy.empty(total_features, dtype=int)
for x in range(total_features):
    arr[x] = x

obs_list = []
Utility.Permutation(arr, total_features, total_features, obs_list)

best_route = {}

for true_idx in range(total_hypos):
    # Create the true hypo
    true_hypo = hypo[true_idx]
    print("True hypothesis = ", true_hypo)

    maximum = 0
    route = obs_list[0]
    for obs in obs_list:
        print("List = ", obs)
        # The observed feature list
        obsd_list = []
        idx = 0
        sum = 0
        # Start to observe
        while idx < total_features:
            obsd_list.append(obs[idx])
            P = Utility.Observe(hypo, true_hypo, obsd_list)
            sum += P
            print("Observed = ", obsd_list, "  P = ", P)
            idx += 1
        if sum > maximum:
            route = obs
            maximum = sum
    best_route[tuple(true_hypo)] = route

print(best_route)

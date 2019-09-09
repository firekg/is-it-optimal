# Tree Model
import numpy
import Utility
import copy
import Generate


# Generate the hypothesis matrix
total_features = 4
hypo = Generate.Boundary_Hypo_Table(total_features, True)

# Create the observation list
arr = numpy.empty(total_features, dtype=int)
for x in range(total_features):
    arr[x] = x

obs_list = []
Utility.Permutation(arr, total_features, total_features, obs_list)
print(obs_list)

for true_idx in range(total_features):
    true_hypo = hypo[1]
    print("True hypothesis = ", true_hypo)

    for obs in obs_list:
        print("List = ", obs)
        # The observed feature list
        obsd_list = []
        idx = 0
        # Start to observe
        while idx < total_features:
            obsd_list.append(obs[idx])
            print("Observed = ", obsd_list, "  P = ",
                  Utility.Observe(hypo, true_hypo, obsd_list))
            idx += 1

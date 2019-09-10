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
arr = ""
for x in range(total_features):
    arr += str(x)

obs_list = numpy.array(list(Utility.Permutation(arr)), dtype=int)
print(obs_list)

best_route={}

# Count how many observations
counting=0

for true_idx in range(total_hypos):
    # Create the true hypo
    true_hypo=hypo[true_idx]
    print("True hypothesis = ", true_hypo)

    maximum=0
    route=[]
    for obs in obs_list:
        print("List = ", obs)
        # The observed feature list
        obsd_list=[]
        idx=0
        sum=0
        # Start to observe
        while idx < total_features:
            obsd_list.append(obs[idx])
            counting += 1
            P=Utility.Observe(hypo, true_hypo, obsd_list)
            print("Observed = ", obsd_list, "  P = ", P)
            sum += P
            if sum + 1 * (total_features - 1 - idx) < maximum:
                print("discarded")
                break
            idx += 1

        # Update the best routes
        if sum == maximum:
            route.append(obs)
        elif sum > maximum:
            route=[obs]
            maximum=sum
    best_route[tuple(true_hypo)]=route

# Report
print("Total observations:", counting)
print(best_route)

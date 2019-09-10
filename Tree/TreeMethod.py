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
lst_size = len(obs_list)
print(obs_list)

best_route = {}

# Count how many observations
counting = 0

for true_idx in range(total_hypos):
    # Create the true hypo
    true_hypo = hypo[true_idx]
    print("True hypothesis = ", true_hypo)

    maximum = 0
    route = []

    list_idx = 0
    while list_idx < lst_size:
        obs = obs_list[list_idx]
        print("List = ", obs)

        # The observed feature list, which is empty at the beginning
        observed_list = []
        idx, sum = 0, 0
        discard = False
        # Start to observe
        while idx < total_features:
            current_feature = obs[idx]
            observed_list.append(current_feature)
            counting += 1
            P = Utility.Observe(hypo, true_hypo, observed_list)
            print("Observed = ", observed_list, "  P = ", P)
            sum += P
            if sum + 1 * (total_features - 1 - idx) < maximum:
                # Stop oberserving the following unobserved feature
                discard = True
                print("discarded")
                a = 1
                while True:
                    if list_idx + a >= lst_size:
                        break
                    elif obs_list[list_idx, idx] != obs_list[list_idx + a, idx]:
                        break
                    else:
                        a += 1
                list_idx += a
                break
            idx += 1
        if not discard:
            # Update the best routes
            if sum == maximum:
                route.append(obs)
            elif sum > maximum:
                route = [obs]
                maximum = sum

            list_idx += 1
    best_route[tuple(true_hypo)] = route

# Report
print("Total observations:", counting)
print(best_route)

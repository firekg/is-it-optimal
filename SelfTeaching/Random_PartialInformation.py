import Generate
import numpy
import copy

num_feature = 4
hypo_table = Generate.Boundary_Hypo_Table(num_feature)
num_hypo = len(hypo_table)

x = numpy.array(range(num_feature))

for a in range(num_feature):
    t = numpy.random.randint(0, num_feature)
    x[t], x[a] = x[a], x[t]
print(x)


observation_steps = 2   # How many observations we want
k_matrix = numpy.zeros((num_hypo, num_hypo))

for tr in range(num_hypo):
    '''True hypothesis'''
    true_hypo = hypo_table[tr]
    print(true_hypo)

    temp = list(range(num_hypo))
    print("temp",temp)
    for idx in range(observation_steps):
        '''Search for the matching'''
        c_idx = x[idx]  # The current feature index
        c_label = true_hypo[c_idx]
        for i in range(num_hypo):
            if c_label != hypo_table[i][c_idx]:
                if i in temp:
                    temp.remove(i)
    
    print(temp)
    for m in temp:
        k_matrix[tr][m] = 1 / len(temp)
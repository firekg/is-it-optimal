import Generate
import numpy
import AL

'''Hyperparameters'''
num_feature = 4     # How many features
observation_steps = 2  # How many observations we want

hypo_table = Generate.Boundary_Hypo_Table(num_feature)
num_hypo = len(hypo_table)
knowledgeability = 1/num_hypo

# Create the instance
task = AL.ActiveLearning()

# Set the environment for the task
task.Set(user_hypo=hypo_table, knowledgeability=knowledgeability)

# task.DS_Task_With_FileOutput()
S = task.P_Task(knowledgeability)

x = []
for ki in S:
    x.append(S[ki])
x = numpy.array(x)

k_matrix = numpy.zeros((num_hypo, num_hypo))

for tr in range(num_hypo):
    '''True hypothesis'''
    true_hypo = hypo_table[tr]
    print(true_hypo)

    temp = list(range(num_hypo))
    print("temp",temp)
    for idx in range(observation_steps):
        '''Search for the matching'''
        c_idx = x[tr][idx]  # The current feature index
        c_label = true_hypo[c_idx]
        for i in range(num_hypo):
            if c_label != hypo_table[i][c_idx]:
                if i in temp:
                    temp.remove(i)
    
    print(temp)
    for m in temp:
        k_matrix[tr][m] = 1 / len(temp)

print(k_matrix)
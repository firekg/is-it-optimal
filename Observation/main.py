import AL

task = AL.ActiveLearning(knowledgeability=1)
task.Set(user_hypo=[[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]])

task.P_Task()

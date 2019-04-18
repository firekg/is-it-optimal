import ALTask

task = ALTask.ActiveLearning(knowledgeability=1.0)
task.Set(user_hypo=[[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
task.K_Task(1000)
print(task.p_learner_h_xy)
task.P_Task()

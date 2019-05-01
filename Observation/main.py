import AL
import Generate

task = AL.ActiveLearning(knowledgeability=1 / 4)
task.Set(user_hypo=Generate.Generate_Boundary_Hypo_Table(3, True))
task.P_Task()

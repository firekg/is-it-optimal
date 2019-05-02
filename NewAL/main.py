import AL
import Generate

task = AL.ActiveLearning(knowledgeability=1 / 7)
task.Set(user_hypo=Generate.Generate_Boundary_Hypo_Table(6, True))
task.P_Task()

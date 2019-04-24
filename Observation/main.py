import AL
import Generate

hypotable = Generate.Generate_Boundary_Hypo_Table(5)
task = AL.ActiveLearning(knowledgeability=1)
task.Set(user_hypo=hypotable)
task.P_Task()

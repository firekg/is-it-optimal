import AL
import Generate

hypotable = Generate.Generate_Boundary_Hypo_Table(10)

task = AL.ActiveLearning(knowledgeability=0.15)
task.Set(user_hypo=hypotable)
task.P_Task(0.1, 1.0, 0.1)

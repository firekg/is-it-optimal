import AL
import Generate

hypotable = Generate.Generate_Uniform_Hypo_Table(5)
print(hypotable)
task = AL.ActiveLearning(Labelsize=2, knowledgeability=0.20)
task.Set(user_hypo=hypotable)
task.P_Task(bottom=0.20, upper=1.00, stepsize=0.20)

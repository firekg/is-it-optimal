import AL
import Generate

hypo = Generate.Zigzag_Hypo_Table(7, False)
task = AL.ActiveLearning(knowledgeability=1 / len(hypo))
task.Set(user_hypo=hypo)
task.P_Task()

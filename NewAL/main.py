import AL
import Generate
import Const

hypo = Generate.Uniform_Hypo_Table(1, False)
task = AL.ActiveLearning(knowledgeability=1)
task.Set(user_hypo=Const.user_hypo_table)
task.O_Task()

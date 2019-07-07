import AL
import Generate
import Const

#hypo = Generate.Uniform_Hypo_Table(1, False)
'''
task = AL.ActiveLearning(knowledgeability=1)

Generate.Transfer_User_Table(Const.user_hypo_table, Const.label_map)
print(Const.user_hypo_table)P
task.Set(user_hypo=Const.user_hypo_table)
task.O_Task()
'''

task = AL.ActiveLearning(knowledgeability=1)
task.Set(user_hypo=Generate.Boundary_Hypo_Table(4,True))
task.DS_Task()
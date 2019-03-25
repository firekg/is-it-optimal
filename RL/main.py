import Task

user_subset = [
      [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]
]
st = Task.Subset_Task(4, 2)
st.Init_Subset(0,user_subset)
st.Simulate(10000)
st.Probability_Map()

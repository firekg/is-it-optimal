# This file defines how state value map will be updated

def MonteCarlo_Update(return_value, state_list, state_value_map):
      for states in state_list:
            if not tuple(states) in state_value_map:
                  state_value_map[tuple(states)] = [1, return_value]
            else:
                  step = state_value_map[tuple(states)][0]
                  value = state_value_map[tuple(states)][1]
                  state_value_map[tuple(states)][1] = (step * value + return_value) / (step + 1)
                  state_value_map[tuple(states)][0] += 1
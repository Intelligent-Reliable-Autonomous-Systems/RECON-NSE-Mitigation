import copy
from timeit import default_timer as timer
from value_iteration_centralized import value_iteration
from init_env_centralized import CentralizedEnvironment
from calculation_lib_centralized import system_do_action
from calculation_lib_centralized import reached_goal


def follow_policy(GRID):
    Pi = copy.copy(GRID.Pi)
    while not reached_goal(GRID):
        R = GRID.R(GRID.js, Pi[GRID.js])
        GRID.Reward += R
        GRID.trajectory.append(((GRID.js[0], GRID.js[1]), Pi[GRID.js], R))
        GRID.plan += " -> " + str(Pi[GRID.js])
        GRID.js = system_do_action(GRID, GRID.js, Pi[GRID.js])
        GRID.path = GRID.path + "->" + str(GRID.js)


Agents = [2]  # , 3, 4, 5]
Goal_Deposit = [(1, 1)]  # , (2, 1), (2, 2), (2, 3)]
Time_keeping = []
for i in range(len(Agents)):
    start_time = timer()
    Grid = CentralizedEnvironment(Agents[i], Goal_Deposit[i], 'train_grid.txt', 'stochastic', 0.8)
    Grid.V, Grid.Pi = value_iteration(Grid, Grid.S, Grid.A, Grid.R, Grid.gamma)
    end_time = timer()
    # follow_policy(Grid)
    # for i in Grid.trajectory:
    #     print(i)
    # print("Reached GOAL: ", Grid.js)
    Time_keeping.append(end_time - start_time)
    print(str(Agents[i]) + " Agents: " + str(Time_keeping[i]) + " sec")

print("Time taken to compute policy using Centralized VI:")
for i in range(len(Agents)):
    print(str(Agents[i]) + " Agents: " + str(Time_keeping[i]) + " sec")

# For 2 x 2 grid
# 1 Agents: 0.30 sec
# 2 Agents: 31.30 sec
# 3 Agents: 2282.94 sec

# For 3 x 3 grid
# 1 Agents: 0.68 sec
# 2 Agents: 155.94 sec
# 3 Agents: 20321.93 sec

import copy

from matplotlib import pyplot as plt

from value_iteration_centralized import value_iteration as VI_c
from init_env_centralized import CentralizedEnvironment
from calculation_lib_centralized import system_do_action
from calculation_lib_centralized import reached_goal
import warnings
from timeit import default_timer as timer
from value_iteration import value_iteration as VI
from init_env import Environment
from init_env import reset_Agents
from display_lib import show_each_agent_plan

from calculation_lib import all_have_reached_goal

# def follow_policy(GRID):
#     Pi = copy.copy(GRID.Pi)
#     while not reached_goal(GRID):
#         R = GRID.R(GRID.js, Pi[GRID.js])
#         GRID.Reward += R
#         GRID.trajectory.append(((GRID.js[0], GRID.js[1]), Pi[GRID.js], R))
#         GRID.plan += " -> " + str(Pi[GRID.js])
#         GRID.js = system_do_action(GRID, GRID.js, Pi[GRID.js])
#         GRID.path = GRID.path + "->" + str(GRID.js)
#
#
# Agents = [2]  # , 3, 4, 5]
# Goal_Deposit = [(1, 1)]  # , (2, 1), (2, 2), (2, 3)]
# Time_keeping = []
# for i in range(len(Agents)):
#     start_time = timer()
#     Grid = CentralizedEnvironment(Agents[i], Goal_Deposit[i], 'train_grid.txt', 'stochastic', 0.8)
#     Grid.V, Grid.Pi = VI_c(Grid, Grid.S, Grid.A, Grid.R, Grid.gamma)
#     end_time = timer()
#     # follow_policy(Grid)
#     # for i in Grid.trajectory:
#     #     print(i)
#     # print("Reached GOAL: ", Grid.js)
#     Time_keeping.append(end_time - start_time)
#     print(str(Agents[i]) + " Agents: " + str(Time_keeping[i]) + " sec")
#
# print("Time taken to compute policy using Centralized VI:")
# for i in range(len(Agents)):
#     print(str(Agents[i]) + " Agents: " + str(Time_keeping[i]) + " sec")

# For 2 x 2 grid
# 1 Agents: 0.30 sec
# 2 Agents: 31.30 sec
# 3 Agents: 2282.94 sec

# For 3 x 3 grid
# 1 Agents: 0.68 sec
# 2 Agents: 155.94 sec
# 3 Agents: 20321.93 sec

Num_Agents = [1, 2, 3]
Goal_Deposit = [(1, 0), (1, 1), (2, 1)]
Time_Keeping_Centralized = [0.68/60.0, 355.94/60.0, 20321.93/60.0]
Time_Keeping_Decentralized = []

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
for i in range(len(Num_Agents)):
    num_of_agents = Num_Agents[i]
    goal_deposit = Goal_Deposit[i]
    mode = 'stochastic'  # 'deterministic' or 'stochastic'
    prob = 0.8

    Grid = Environment(num_of_agents, goal_deposit, "train_grid.txt", mode, prob)

    # initialize agents with the initial coordinating policies
    start_timer = timer()
    Agents = Grid.init_agents_with_initial_policy()
    end_timer = timer()
    Time_Keeping_Decentralized.append((end_timer - start_timer)/60.0)

print("Decentralized Simulation:")
for i in range(len(Num_Agents)):
    print(str(Num_Agents[i]) + " Agents: " + str(Time_Keeping_Decentralized[i]) + " sec")

fig, ax = plt.subplots()

ax.set_xlabel('Number of Agents')
plt.xticks(Num_Agents, Num_Agents)
ax.set_ylabel('Time (min)')
ax.set_title('Centralized Time vs Decentralized Time\nwith number of agents')

plt.plot(Num_Agents, Time_Keeping_Centralized, color='r', label='Centralized')
plt.plot(Num_Agents, Time_Keeping_Decentralized, color='b', label='Decentralized')

ax.legend()
plt.show()

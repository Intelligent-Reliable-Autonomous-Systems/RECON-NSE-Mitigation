import numpy as np
from timeit import default_timer as timer
from blame_assignment import Blame
import init_env
from init_env import Environment, take_step
from init_env import get_joint_state, get_reward_by_following_policy
from init_agent import Agent
import value_iteration
from display_lib import display_values, display_policy, display_grid_layout
from display_lib import display_agent_log, display_just_grid
from blame_assignment import generate_counterfactuals, get_joint_NSEs_for_list
from calculation_lib import all_have_reached_goal
import copy
from heapq import nsmallest

# from blame_assigment import Blame
# from remove_NSE import NSE_action_ban

rows = init_env.rows
columns = init_env.columns

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
M = 1
num_of_agents = 2

# initialize the environment
trash_repository = {'S': 1, 'L': 1}
Grid = Environment(trash_repository, num_of_agents)
# initialize agents
Agents = []
for label in range(num_of_agents):
    Agents.append(Agent((0, 0), Grid, str(label + 1)))

# value iteration for all agents
for agent in Agents:
    agent.V, agent.Pi = value_iteration.value_iteration(agent, Grid.S)
for agent in Agents:
    # print("=====================   Agent " + str(agent.label) + "   =====================")
    for s in agent.Pi:
        act = agent.Pi[s]
        # print(str(s) + ": [" + str(act) + "] --> reward: " + str(agent.Reward(s, act)))
    # print("=====================================================\n")
for agent in Agents:
    agent.s = copy.deepcopy(agent.s0)
    agent.follow_policy()
    # print("==================== Trajectory for Agent " + agent.label + "  =====================")
    # for sar in Agents[0].trajectory:
    #     print(sar)
    # print("==================================================================\n")
print("Environment:")
display_just_grid(Grid.All_States)
for agent in Agents:
    print("Plan for Agent " + agent.label + ":")
    print(agent.plan[4:])  # starting for 4 to avoid the initial arrow display ' -> '
    print("________________________________________________\n")

for agent in Agents:
    agent.agent_reset()

path_joint_states = [get_joint_state(Agents)]  # Store the starting joint states
path_joint_NSE_values = [
    Grid.give_joint_NSE_value(get_joint_state(Agents))]  # Store the corresponding joint NSE
joint_NSE_states = []
joint_NSE_values = []

while all_have_reached_goal(Agents) is False:
    Agents, joint_NSE = take_step(Grid, Agents)
    joint_state = get_joint_state(Agents)
    path_joint_states.append(joint_state)
    path_joint_NSE_values.append(joint_NSE)
    if joint_NSE != 0:
        joint_NSE_states.append(joint_state)
        joint_NSE_values.append(joint_NSE)

print('NSE Report:')
for x in range(len(path_joint_states)):
    print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))

R_old = 0  # Just for storage purposes
NSE_old = 0  # Just for storage purposes
if all_have_reached_goal(Agents):
    print("\nAll Agents have reached the GOAL!!!")
    R_old = round(float(np.sum([agent.R for agent in Agents])), 2)
    NSE_old = round(float(np.sum(path_joint_NSE_values[:])), 2)
    print("Total Reward: ", R_old)
    print("Total NSE: ", NSE_old)
print('______________________________________\nAgent Logs:')

# displaying individual logs of agents -  path, Reward, actual NSE contribution
for agent in Agents:
    print("\nAgent " + agent.label + ":")
    display_agent_log(agent, 'before')
print('______________________________________')

agentWise_cfs = []
start_timer = timer()
for js_nse in joint_NSE_states:
    agentWise_cfs, cfs = generate_counterfactuals(js_nse, Grid, Agents)
    print("\n==== in joint state === : " + str(js_nse) + ": " + str(Grid.give_joint_NSE_value(js_nse)))
    for i in range(len(agentWise_cfs)):
        print("CounterFactual states for Agent " + str(i + 1))
        for cf in agentWise_cfs[i]:
            cf_nse = Grid.give_joint_NSE_value(cf)
            print(str(cf) + ": " + str(cf_nse))

end_timer = timer()
print("-------------------------------------------------------")
print("Time taken for generating counterfactuals: " + str(
    round((end_timer - start_timer) * 1000, 3)) + " ms")
print("-------------------------------------------------------")

print("============== BLAME ASSIGNMENT ==============")
blame = Blame(Agents, Grid)
blame_distribution = {}  # dict to store blame distributions of joint states as an array [A1_blame, A2_blame,...]
for js_nse in joint_NSE_states:
    print("-------\nBlame distribution for " + str(js_nse) + " =")  # , end="")
    original_NSE = Grid.give_joint_NSE_value(js_nse)
    blame_values = blame.get_blame(original_NSE, js_nse)
    blame_distribution[js_nse] = np.around(blame_values, 2)
    print("\t" + str(round(original_NSE, 2)) + " : " + str(np.around(blame_values, 2)))

for js_nse in joint_NSE_states:
    for agent_idx in range(len(Agents)):
        agent = Agents[agent_idx]
        blame_array_for_js = -blame_distribution[js_nse]
        s = js_nse[agent_idx]
        agent.R_blame[(s[0], s[1], s[3])] = blame_array_for_js[agent_idx]

exit(0)  # temporary stopping the code to analyse policy

# Loop for computing R_blame, printing it, and resetting all agents
for agent in Agents:
    agent = Grid.add_goal_reward(agent)
    print("--------------\nR_blame for Agent ", agent.label)
    agent.R = 0
    agent.s = copy.deepcopy(agent.s0)
    display_values(agent.R_blame)

# getting R_blame rewards for each agent by re-simulating original policies
agent_Rblame_dict = get_reward_by_following_policy(Agents)
print(agent_Rblame_dict)

agent_labels_to_be_corrected = nsmallest(M, agent_Rblame_dict, key=agent_Rblame_dict.get)
print("Agents to be corrected: ", agent_labels_to_be_corrected)
Agents_to_be_corrected = [Agents[int(i) - 1] for i in agent_labels_to_be_corrected]

print("\n---- Now doing Lexicographic Value Iteration for selected agents ----\n ")
for agent in Agents_to_be_corrected:
    agent = value_iteration.action_set_value_iteration(agent, Grid.S, Grid.R, agent.gamma)
    agent.V, agent.Pi = value_iteration.value_iteration(agent, Grid.S, agent.R_blame, agent.gamma)

# for agent in Agents_to_be_corrected:
#     print("Final Policy for Agent " + str(agent.label) + ": ")
#     display_policy(agent.Pi)

for agent in Agents:
    agent.agent_reset()

path_joint_states = [get_joint_state(Agents)]
path_joint_NSE_values = [Grid.give_joint_NSE_value(get_joint_state(Agents))]
joint_NSE_states = []
joint_NSE_values = []
# print(Grid.R)

while not all_have_reached_goal(Agents):
    Agents, joint_NSE = take_step(Grid, Agents)
    joint_state = get_joint_state(Agents)
    path_joint_states.append(joint_state)
    path_joint_NSE_values.append(joint_NSE)
    if joint_NSE != 0:
        joint_NSE_states.append(joint_state)
        joint_NSE_values.append(joint_NSE)

print("Policy Agent 1:")
display_policy(agent1.Pi)
print("Policy Agent 2:")
display_policy(agent2.Pi)
# NEW CORRECTED Joint NSE display log of all joint states and NSEs reported
print("\n\nNEW Joint State NSE report:")
for i in range(len(path_joint_states)):
    print(str(path_joint_states[i]) + ": " + str(path_joint_NSE_values[i]))

R_new = round(float(np.sum([agent.R for agent in Agents])), 2)
NSE_new = round(float(np.sum(path_joint_NSE_values[:])), 2)

if all_have_reached_goal(Agents):
    print("\nAll Agents have reached the GOAL!!\nWe corrected (" + str(M) + "/" + str(len(Agents)) + ") Agents:")
    print("Corrected Agent list: ", agent_labels_to_be_corrected)
    print("--------------------------")
    print("OLD Total Reward: ", R_old)
    print("OLD Total NSE: ", NSE_old)
    print("--------------------------")
    print("NEW Total Reward: ", R_new)
    print("NEW Total NSE: ", NSE_new)
    print("--------------------------")
print("NSE range: ", blame.NSE_window)

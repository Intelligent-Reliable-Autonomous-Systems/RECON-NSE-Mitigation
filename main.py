import numpy as np
from timeit import default_timer as timer
from blame_assignment import Blame
import init_env
from init_env import Environment, take_step
from init_env import get_joint_state, get_reward_by_following_policy
from init_agent import Agent
import value_iteration
from display_lib import display_values, display_policy, display_grid_layout
from display_lib import display_agent_log
from blame_assignment import generate_counterfactuals, get_joint_NSEs_for_list
from calculation_lib import all_have_reached_goal
import copy
from heapq import nsmallest

# from blame_assigment import Blame
# from remove_NSE import NSE_action_ban

rows = init_env.rows
columns = init_env.columns

# Number of agent to be corrected (M = 2)/(out of N = 5)
M = 3

agent1_startLoc = (0, 0)
agent2_startLoc = (4, 0)
agent3_startLoc = (0, 5)
agent4_startLoc = (6, 7)
agent5_startLoc = (7, 3)
agent6_startLoc = (0, 7)
agent7_startLoc = (7, 0)
agent8_startLoc = (6, 0)

# initialize the environment
trash_repository = {'S': 5, 'M': 5, 'L': 5}
Grid = Environment(trash_repository)
# initialize agent
agent1 = Agent(agent1_startLoc, Grid, '1')
agent2 = Agent(agent2_startLoc, Grid, '2')
agent3 = Agent(agent3_startLoc, Grid, '3')
agent4 = Agent(agent4_startLoc, Grid, '4')
agent5 = Agent(agent5_startLoc, Grid, '5')
# agent6 = Agent(agent6_startLoc, Grid, 'M', '6')
# agent7 = Agent(agent7_startLoc, Grid, 'L', '7')
# agent8 = Agent(agent8_startLoc, Grid, 'S', '8')
Agents = [agent1, agent2, agent3, agent4, agent5]  # , agent6, agent7, agent8]


# updating trash repository by removing selected options by agents
for agent in Agents:
    Grid.trash_repository[agent.s[2]] -= 1
    if Grid.trash_repository[agent.s[2]] < 0:
        display_message = "!!!TRASH ERROR!!! -> junk of size " + agent.s[
            2] + " is not available for agent " + agent.label + "!!"
        print("\n", display_message)
        exit()

# value iteration for all agents
for agent in Agents:
    agent.V, agent.Pi = value_iteration.value_iteration(agent, Grid.S, Grid.R, agent.gamma)

path_joint_states = [get_joint_state(Agents)]  # Store the starting joint states
path_joint_NSE_values = [
    Grid.give_joint_NSE_value(get_joint_state(Agents))]  # Store the corresponding joint NSE
joint_NSE_states = []
joint_NSE_values = []

while not all_have_reached_goal(Agents):
    Agents, joint_NSE = take_step(Grid, Agents)
    joint_state = get_joint_state(Agents)
    path_joint_states.append(joint_state)
    path_joint_NSE_values.append(joint_NSE)
    if joint_NSE != 0:
        joint_NSE_states.append(joint_state)
        joint_NSE_values.append(joint_NSE)

print("\nGrid:")
display_grid_layout(init_env.all_states, Agents)
print("Policy:")
display_policy(agent1.Pi)

print("Agent spawn locations:")
for agent in Agents:
    print("Agent " + agent.label + " started at " + str(agent.startLoc) + " and is now at " + str(
        agent.Pi[(agent.s[0], agent.s[1], agent.s[3])]))

R_old = 0  # Just for storage purposes
NSE_old = 0  # Just for storage purposes
if all_have_reached_goal(Agents):
    print("\nAll Agents have reached the GOAL!!!")
    R_old = round(float(np.sum([agent.R for agent in Agents])), 2)
    NSE_old = round(float(np.sum(path_joint_NSE_values[:])), 2)
    print("Total Reward: ", R_old)
    print("Total NSE: ", NSE_old)
    print("\n")

# displaying individual logs of agents -  path, Reward, actual NSE contribution
for agent in Agents:
    print("\nAgent " + agent.label + ":")
    display_agent_log(agent, 'before')

# Actual Joint NSE display log of all joint states and NSEs reported
print("\n\n=================\nJoint State NSE report:")
for i in range(len(path_joint_states)):
    print(str(path_joint_states[i]) + ": " + str(path_joint_NSE_values[i]))

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
print("\n-------------------\nTime taken for generating counterfactuals: " + str(
    round((end_timer - start_timer) * 1000, 3)) + " ms")
print("==============================================")
print("Trash Repo: ", trash_repository)
print("==============================================")
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

import numpy as np
from timeit import default_timer as timer
from blame_assignment import Blame
import init_env
from init_env import Environment, take_step
from init_env import get_joint_state, get_blame_reward_by_following_policy
from init_agent import Agent
import value_iteration
from display_lib import display_values, display_policy, display_grid_layout
from display_lib import display_agent_log, display_just_grid
from display_lib import plot_reward_bar_comparisons, plot_NSE_bar_comparisons
from blame_assignment import generate_counterfactuals, get_joint_NSEs_for_list
from calculation_lib import all_have_reached_goal
import copy
from heapq import nsmallest, nlargest

# from blame_assigment import Blame
# from remove_NSE import NSE_action_ban

rows = init_env.rows
columns = init_env.columns

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
M = 3
num_of_agents = 3

# initialize the environment
trash_repository = {'S': 1, 'L': 1}
Complete_sim_start_timer = timer()
Grid = Environment(trash_repository, num_of_agents)
# initialize agents
Agents = []
for label in range(num_of_agents):
    Agents.append(Agent((0, 0), Grid, str(label + 1)))

# value iteration for all agents
for agent in Agents:
    agent.V, agent.Pi = value_iteration.value_iteration(agent, Grid.S)

for agent in Agents:
    agent.s = copy.deepcopy(agent.s0)
    agent.follow_policy()

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
    R_old = [round(agent.R, 2) for agent in Agents]
    NSE_old = round(float(np.sum(path_joint_NSE_values)), 2)
    print("Total Reward: ", sum(R_old))
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
    agentWise_cfs, _ = generate_counterfactuals(js_nse, Agents)
    print("\n==== in joint state === : " + str(js_nse) + ": " + str(Grid.give_joint_NSE_value(js_nse)))
    for i in range(len(agentWise_cfs)):
        print("CounterFactual states for Agent " + str(i + 1))
        for cf in agentWise_cfs[i]:
            cf_nse = Grid.give_joint_NSE_value(cf)
            print(str(cf) + ": " + str(cf_nse))

end_timer = timer()
cf_generation_time = round((end_timer - start_timer) * 1000, 3)
print("-------------------------------------------------------")
print("Time taken for generating counterfactuals: " + str(cf_generation_time) + " ms")
print("-------------------------------------------------------")

print("============== BLAME ASSIGNMENT ==============")
blame = Blame(Agents, Grid)
blame_distribution = {}  # dict to store blame distributions of joint states as an array [A1_blame, A2_blame,...]
weighting = {'X': 0.0, 'S': 3.0, 'L': 10.0}
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
        agent.R_blame[s] = blame_array_for_js[agent_idx]

blame.get_training_data(Agents, joint_NSE_states)

# Loop for computing R_blame, printing it, and resetting all agents
for agent in Agents:
    agent = Grid.add_goal_reward(agent)
    print("--------------\nR_blame for Agent ", agent.label)
    agent.R = 0
    agent.s = copy.deepcopy(agent.s0)
    # display_values(agent.R_blame)
    print("Training data for Agent ", agent.label)
    for i in range(len(agent.blame_training_data_x)):
        print(str(agent.blame_training_data_x[i]) + ": " + str(agent.blame_training_data_y[i]))

for agent in Agents:
    agent.generalize_Rblame_linearReg()
    agent.agent_reset()

# exit(0)
# getting R_blame rewards for each agent by re-simulating original policies
NSE_blame_per_agent_before_mitigation = get_blame_reward_by_following_policy(Agents)
print("\n--------Agent-wise NSE (before mitigation) --------\n", NSE_blame_per_agent_before_mitigation)
agent_labels_to_be_corrected = [NSE_blame_per_agent_before_mitigation.index(x) + 1 for x in
                                sorted(NSE_blame_per_agent_before_mitigation, reverse=True)[:M]]
print("\nAgents to be corrected: ", agent_labels_to_be_corrected)
Agents_to_be_corrected = [Agents[int(i) - 1] for i in agent_labels_to_be_corrected]

for agent in Agents:
    agent.agent_reset()

print("\n---- Now doing Lexicographic Value Iteration with R_blame for selected agents ----\n ")
for agent in Agents_to_be_corrected:
    agent = value_iteration.action_set_value_iteration(agent, Grid.S)
    agent.V, agent.Pi = value_iteration.blame_value_iteration(agent, Grid.S, agent.R_blame)

for agent in Agents:
    agent.s = copy.deepcopy(agent.s0)
    agent.follow_policy()

print("Environment:")
display_just_grid(Grid.All_States)
for agent in Agents:
    print("Corrected Plan for Agent " + agent.label + ":")
    print(agent.plan[4:])  # starting for 4 to avoid the initial arrow display ' -> '
    print("________________________________________________\n")

for agent in Agents:
    agent.agent_reset()

path_joint_states = [get_joint_state(Agents)]
path_joint_NSE_values = [Grid.give_joint_NSE_value(get_joint_state(Agents))]
joint_NSE_states = []
joint_NSE_values = []
# print(Grid.R)


while all_have_reached_goal(Agents) is False:
    Agents, joint_NSE = take_step(Grid, Agents)
    joint_state = get_joint_state(Agents)
    path_joint_states.append(joint_state)
    path_joint_NSE_values.append(joint_NSE)
    if joint_NSE != 0:
        joint_NSE_states.append(joint_state)
        joint_NSE_values.append(joint_NSE)

blame_distribution_stepwise = []
start_timer_for_blame_without_gen = timer()
for js_nse in joint_NSE_states:
    original_NSE = Grid.give_joint_NSE_value(js_nse)
    blame_values = blame.get_blame(original_NSE, js_nse)
    blame_distribution_stepwise.append(blame_values)
end_timer_for_blame_without_gen = timer()
time_for_blame_without_gen = round((end_timer_for_blame_without_gen - start_timer_for_blame_without_gen) * 1000, 3)

print('NSE Report (after R_blame):')
for x in range(len(path_joint_states)):
    print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))

R_new = 0
NSE_new = 0
if all_have_reached_goal(Agents):
    print("-----------------------------------")
    R_new = [round(agent.R, 2) for agent in Agents]
    NSE_new = round(float(np.sum(path_joint_NSE_values)), 2)
    print("Total Reward (after R_blame): ", R_new)
    print("Total NSE (after R_blame): ", NSE_new)
print('-----------------------------------')

NSE_blame_per_agent_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
print("\n--------Agent-wise NSE (with R_blame mitigation) --------\n", NSE_blame_per_agent_after_R_blame)
print("")
for agent in Agents:
    agent.agent_reset()

print("\n---- Now doing Lexicographic Value Iteration with R_blame_gen for selected agents ----\n ")
for agent in Agents_to_be_corrected:
    agent = value_iteration.action_set_value_iteration(agent, Grid.S)
    agent.V, agent.Pi = value_iteration.blame_value_iteration(agent, Grid.S, agent.R_blame_gen)

for agent in Agents:
    agent.s = copy.deepcopy(agent.s0)
    agent.follow_policy()

print("Environment:")
display_just_grid(Grid.All_States)
for agent in Agents:
    print("Corrected Plan for Agent " + agent.label + ":")
    print(agent.plan[4:])  # starting for 4 to avoid the initial arrow display ' -> '
    print("________________________________________________\n")

for agent in Agents:
    agent.agent_reset()

path_joint_states = [get_joint_state(Agents)]
path_joint_NSE_values = [Grid.give_joint_NSE_value(get_joint_state(Agents))]
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

print('NSE Report (after R_blame_gen):')
for x in range(len(path_joint_states)):
    print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))

blame_distribution_stepwise = []
start_timer_for_blame_with_gen = timer()
for js_nse in joint_NSE_states:
    original_NSE = Grid.give_joint_NSE_value(js_nse)
    blame_values = blame.get_blame(original_NSE, js_nse)
    blame_distribution_stepwise.append(blame_values)
end_timer_for_blame_with_gen = timer()
time_for_blame_with_gen = round((end_timer_for_blame_with_gen - start_timer_for_blame_with_gen) * 1000, 3)

NSE_blame_per_agent_after_R_blame_gen = [sum(x) for x in zip(*blame_distribution_stepwise)]
print("\n--------Agent-wise NSE (with R_blame_gen mitigation) --------\n", NSE_blame_per_agent_after_R_blame_gen)
R_new_gen = []
if all_have_reached_goal(Agents):
    print("-----------------------------------")
    print("Total Reward (initial): ", R_old)
    print("Total NSE (initial): ", NSE_old)
    print("-----------------------------------")
    print("Total Reward (after R_blame): ", R_new)
    print("Total NSE (after R_blame): ", NSE_new)
    print("-----------------------------------")
    R_new_gen = [round(agent.R, 2) for agent in Agents]
    NSE_new_gen = round(float(np.sum(path_joint_NSE_values[:])), 2)
    print("Total Reward (after R_blame_gen): ", R_new_gen)
    print("Total NSE (after R_blame_gen): ", NSE_new_gen)
print('-----------------------------------')

Complete_sim_end_timer = timer()
Complete_sim_time = round((Complete_sim_end_timer - Complete_sim_start_timer), 3)
print("-------------------- TIME KEEPING ----------------------")
print("Complete Simulation: " + str(Complete_sim_time) + " sec")
print("CF generation for Blame Assignment (without Generalization): " + str(time_for_blame_without_gen) + " ms")
print("CF generation for Blame Assignment (with Generalization): " + str(time_for_blame_with_gen) + " ms")
print("-------------------------------------------------------")
##########################################
########### PLOTTING SECTION #############
##########################################

plot_reward_bar_comparisons(R_old, R_new, R_new_gen)
plot_NSE_bar_comparisons(NSE_blame_per_agent_before_mitigation, NSE_blame_per_agent_after_R_blame,
                         NSE_blame_per_agent_after_R_blame_gen)

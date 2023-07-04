import warnings
import numpy as np
from timeit import default_timer as timer
import random
from blame_assignment import Blame
from init_env import Environment, get_total_R_and_NSE_from_path
from init_env import reset_Agents, show_joint_states_and_NSE_values
import value_iteration
from display_lib import display_just_grid, plot_NSE_bar_comparisons_with_std_mean
from display_lib import plot_reward_bar_comparisons, plot_blame_bar_comparisons
from display_lib import plot_NSE_bar_comparisons, show_each_agent_plan

from calculation_lib import all_have_reached_goal

warnings.filterwarnings('ignore')

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
M = 2
num_of_agents = 2
goal_deposit = (1, 1)

# Tracking NSE values with grids
NSE_old_tracker = []
NSE_new_tracker = []
NSE_new_gen_tracker = []
num_of_agents_tracker = []

# initialize the environment
Complete_sim_start_timer = timer()
Grid = Environment(num_of_agents, goal_deposit, "grids/train_grid.txt")

# initialize agents with the initial coordinating policies
Agents = Grid.init_agents_with_initial_policy()

# displaying grid for visual
# display_just_grid(Grid.All_States)
show_each_agent_plan(Agents)
print("Agent 1: \n", Agents[0].path)
print("Agent 2: \n", Agents[1].path)
Agents = reset_Agents(Agents)

joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report:')
R_old, NSE_old = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
# display_all_agent_logs(Agents)
joint_NSE_states_training = joint_NSE_states
# BLAME ASSIGNMENT begins now
blame = Blame(Agents, Grid)
blame_training = blame
blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
blame.get_training_data(Agents, joint_NSE_states)
Agents = reset_Agents(Agents)

for agent in Agents:
    agent.generalize_Rblame_linearReg()

# getting R_blame rewards for each agent by re-simulating original policies
NSE_blame_per_agent_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
print("\n--------Agent-wise NSE blames (before mitigation) --------\n", NSE_blame_per_agent_before_mitigation)
sorted_indices = sorted(range(len(NSE_blame_per_agent_before_mitigation)),
                        key=lambda a: NSE_blame_per_agent_before_mitigation[a], reverse=True)
top_values = [NSE_blame_per_agent_before_mitigation[i] for i in sorted_indices[:3]]
top_offender_agents = [i + 1 for i, value in enumerate(NSE_blame_per_agent_before_mitigation) if value in top_values]
print("\nAgents to be corrected: ", top_offender_agents)
Agents_to_be_corrected = [Agents[i - 1] for i in top_offender_agents]

Agents = reset_Agents(Agents)

print("\n---- Now doing Lexicographic Value Iteration with R_blame for selected agents ----\n ")
value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame')
joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report (R_blame):')

blame_distribution_stepwise = []
start_timer_for_blame_without_gen = timer()
for js_nse in joint_NSE_states:
    original_NSE = Grid.give_joint_NSE_value(js_nse)
    blame_values = blame.get_blame(original_NSE, js_nse)
    blame_distribution_stepwise.append(blame_values)
end_timer_for_blame_without_gen = timer()
time_for_blame_without_gen = round((end_timer_for_blame_without_gen - start_timer_for_blame_without_gen) * 1000, 3)

R_new, NSE_new = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
# display_all_agent_logs(Agents)

NSE_blame_per_agent_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
print("\n--------Agent-wise NSE (with R_blame mitigation) --------\n", NSE_blame_per_agent_after_R_blame)

Agents = reset_Agents(Agents)

print("\n---- Now doing Lexicographic Value Iteration with R_blame_gen for selected agents ----\n ")
value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen')
joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report (R_blame_gen):')

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
R_new_gen, NSE_new_gen = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

if all_have_reached_goal(Agents):
    print("-----------------------------------------------")
    print("Total Reward (initial): ", R_old)
    print("Total NSE (initial): ", NSE_old)
    print("-----------------------------------------------")
    print("Total Reward (after R_blame): ", R_new)
    print("Total NSE (after R_blame): ", NSE_new)
    print("-----------------------------------------------")
    print("Total Reward (after R_blame_gen): ", R_new_gen)
    print("Total NSE (after R_blame_gen): ", NSE_new_gen)
    print("-----------------------------------------------")

NSE_old_tracker.append(NSE_old)
NSE_new_tracker.append(NSE_new)
NSE_new_gen_tracker.append(NSE_new_gen)
num_of_agents_tracker.append(num_of_agents)

Complete_sim_end_timer = timer()
Complete_sim_time = round((Complete_sim_end_timer - Complete_sim_start_timer), 3)
print("-------------------- TIME KEEPING ----------------------")
print("Complete Simulation: " + str(Complete_sim_time) + " sec")
print("CF generation for Blame Assignment (without Generalization): " + str(time_for_blame_without_gen) + " ms")
print("CF generation for Blame Assignment (with Generalization): " + str(time_for_blame_with_gen) + " ms")
print("-------------------------------------------------------")

##########################################
#           PLOTTING SECTION             #
##########################################

plot_reward_bar_comparisons(R_old, R_new, R_new_gen, Grid)
plot_blame_bar_comparisons(NSE_blame_per_agent_before_mitigation, NSE_blame_per_agent_after_R_blame,
                           NSE_blame_per_agent_after_R_blame_gen, Grid)
plot_NSE_bar_comparisons(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_tracker, num_of_agents_tracker, Grid)
print("Num of agents array = ", num_of_agents_tracker)

exit(0)
##########################################
#            TESTING SECTION             #
##########################################

for i in [str(x) for x in range(1, 6)]:
    filename = 'grids/test_grid' + i + '.txt'
    print("======= Now in test_grid" + i + ".txt =======")
    Grid = Environment(num_of_agents, goal_deposit, filename)

    # initialize agents with the initial coordinating policies
    Agents = Grid.init_agents_with_initial_policy()
    Agents = reset_Agents(Agents)

    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_old, NSE_old = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    # BLAME ASSIGNMENT begins NOW
    blame = Blame(Agents, Grid)
    blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    blame_training.get_training_data(Agents, joint_NSE_states_training)

    Agents = reset_Agents(Agents)
    for agent in Agents:
        agent.generalize_Rblame_linearReg()

    # getting R_blame rewards for each agent by re-simulating original policies
    NSE_blame_per_agent_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
    sorted_indices = sorted(range(len(NSE_blame_per_agent_before_mitigation)),
                            key=lambda a: NSE_blame_per_agent_before_mitigation[a], reverse=True)
    top_values = [NSE_blame_per_agent_before_mitigation[i] for i in sorted_indices[:3]]
    top_offender_agents = [i + 1 for i, value in enumerate(NSE_blame_per_agent_before_mitigation) if
                           value in top_values]
    Agents_to_be_corrected = [Agents[i - 1] for i in top_offender_agents]
    Agents = reset_Agents(Agents)

    # Lexicographic Value Iteration with R_blame for selected agents
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report (R_blame):')
    R_new, NSE_new = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    NSE_blame_per_agent_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
    Agents = reset_Agents(Agents)

    # Lexicographic Value Iteration with R_blame_gen for selected agents
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    NSE_blame_per_agent_after_R_blame_gen = [sum(x) for x in zip(*blame_distribution_stepwise)]
    R_new_gen, NSE_new_gen = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    NSE_old_tracker.append(NSE_old)
    NSE_new_tracker.append(NSE_new)
    NSE_new_gen_tracker.append(NSE_new_gen)
    num_of_agents_tracker.append(num_of_agents)

print("NSE_old_tracker: ", NSE_old_tracker)
print("NSE_new_tracker: ", NSE_new_tracker)
print("NSE_new_gen_tracker: ", NSE_new_gen_tracker)

plot_NSE_bar_comparisons_with_std_mean(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_tracker)

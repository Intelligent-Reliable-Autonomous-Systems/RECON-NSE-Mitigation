import warnings
import numpy as np
import simple_colors
from timeit import default_timer as timer
import display_lib
import value_iteration
from blame_assignment import Blame
from init_env import Environment, get_total_R_and_NSE_from_path
from init_env import reset_Agents, show_joint_states_and_NSE_values
from display_lib import plot_NSE_bar_comparisons, plot_NSE_bars_with_num_agents, show_each_agent_plan

from calculation_lib import all_have_reached_goal

warnings.filterwarnings('ignore')

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
MM = [1, 2, 2, 3, 4, 6, 12]
Num_of_agents = [2, 3, 4, 5, 8, 12, 20]
Goal_deposit = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 4), (4, 8), (8, 12)]

# Tracking NSE values with grids
NSE_old_tracker = []
NSE_new_tracker = []
NSE_new_gen_wo_cf_tracker = []
NSE_new_gen_with_cf_tracker = []
num_of_agents_tracker = []
time_tracker = []
initial_policy_time_tracker = []
LVI_time_tracker = []
LVI_wo_cf_time_tracker = []
LVI_w_cf_time_tracker = []

for ctr in range(0, len(MM)):

    M = MM[ctr]
    num_of_agents = Num_of_agents[ctr]
    goal_deposit = Goal_deposit[ctr]

    print("------------------------------------------")
    print("Number of Agents: ", num_of_agents)
    print("Number of Agents to be corrected: ", M)
    print("Goal deposit: ", goal_deposit)

    mode = 'stochastic'  # 'deterministic' or 'stochastic'
    prob = 0.8
    # initialize the environment
    Complete_sim_start_timer = timer()
    Grid = Environment(num_of_agents, goal_deposit, "grids/train_grid.txt", mode, prob)

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
    blame.get_training_data_wo_cf(Agents, joint_NSE_states)
    blame.get_training_data_with_cf(Agents, joint_NSE_states)
    Agents = reset_Agents(Agents)

    for agent in Agents:
        agent.generalize_Rblame_linearReg()

    # getting R_blame rewards for each agent by re-simulating original policies
    blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
    print("\n--------Agent-wise NSE blames (before mitigation) --------\n" + str(
        blame_before_mitigation) + " = " + simple_colors.red(str(sum(blame_before_mitigation)), ['bold']))
    sorted_indices = sorted(range(len(blame_before_mitigation)),
                            key=lambda a: blame_before_mitigation[a], reverse=True)
    Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
    print("\nAgents to be corrected: ", Agents_for_correction)
    Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

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

    blame_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame mitigation) --------\n" + str(
        blame_after_R_blame) + " = " + simple_colors.magenta(str(sum(blame_after_R_blame)),
                                                             ['bold']))

    Agents = reset_Agents(Agents)

    print("\n---- Now doing Lexicographic Value Iteration with R_blame_gen_wo_cf for selected agents ----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_wo_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_wo_cf):')

    blame_distribution_stepwise = []
    start_timer_for_blame_with_gen = timer()
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)
    end_timer_for_blame_with_gen = timer()
    time_for_blame_with_gen = round((end_timer_for_blame_with_gen - start_timer_for_blame_with_gen) * 1000, 3)

    blame_after_R_blame_gen_wo_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame_gen mitigation) --------\n" + str(
        blame_after_R_blame_gen_wo_cf) + " = " + simple_colors.blue(str(sum(blame_after_R_blame_gen_wo_cf)), ['bold']))
    R_new_gen_wo_cf, NSE_new_gen_wo_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    Agents = reset_Agents(Agents)

    print("\n---- Now doing Lexicographic Value Iteration with R_blame_gen_with_cf for selected agents ----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_with_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_with_cf):')

    blame_distribution_stepwise = []
    start_timer_for_blame_with_gen = timer()
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)
    end_timer_for_blame_with_gen = timer()
    time_for_blame_with_gen = round((end_timer_for_blame_with_gen - start_timer_for_blame_with_gen) * 1000, 3)

    blame_after_R_blame_gen_with_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n-----Agent-wise NSE (with R_blame_gen mitigation) ------\n" + str(
        blame_after_R_blame_gen_with_cf) + " = " + simple_colors.green(str(sum(blame_after_R_blame_gen_with_cf)),
                                                                       ['bold']))
    R_new_gen_with_cf, NSE_new_gen_with_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    if all_have_reached_goal(Agents):
        print("----------------------------------------------")
        print("Total Reward (initial): ", R_old)
        print("Total NSE (initial): ", simple_colors.red(str(NSE_old), ['bold']))
        print("----------------------------------------------")
        print("Total Reward (after R_blame): ", R_new)
        print("Total NSE (after R_blame): ", simple_colors.magenta(str(NSE_new), ['bold']))
        print("----------------------------------------------")
        print("Total Reward (after R_blame_gen_wo_cf): ", R_new_gen_wo_cf)
        print("Total NSE (after R_blame_gen_wo_cf): ", simple_colors.blue(str(NSE_new_gen_wo_cf), ['bold']))
        print("----------------------------------------------")
        print("Total Reward (after R_blame_gen_with_cf): ", R_new_gen_with_cf)
        print("Total NSE (after R_blame_gen_with_cf): ", simple_colors.green(str(NSE_new_gen_with_cf), ['bold']))
        print("----------------------------------------------")

    NSE_old_tracker.append(NSE_old)
    NSE_new_tracker.append(NSE_new)
    NSE_new_gen_wo_cf_tracker.append(NSE_new_gen_wo_cf)
    NSE_new_gen_with_cf_tracker.append(NSE_new_gen_with_cf)
    num_of_agents_tracker.append(num_of_agents)

    Complete_sim_end_timer = timer()
    Complete_sim_time = round((Complete_sim_end_timer - Complete_sim_start_timer), 3)

    # print("-------------------- TIME KEEPING ----------------------")
    # print("Complete Simulation: " + str(Complete_sim_time) + " sec")
    # print("CF generation for Blame Assignment (without Generalization): " + str(time_for_blame_without_gen) + " ms")
    # print("CF generation for Blame Assignment (with Generalization): " + str(time_for_blame_with_gen) + " ms")
    # print("-------------------------------------------------------")

    time_tracker.append(Complete_sim_time)

for i in range(len(time_tracker)):
    time_tracker[i] = round(time_tracker[i] / 60, 2)
    print(str(Num_of_agents[i]) + " Agents: " + str(time_tracker[i]) + " min.")
    print("-------------------------------------------------------")
# display_lib.time_plot(num_of_agents_tracker, time_tracker)
# display_lib.separated_time_plot(num_of_agents_tracker, initial_policy_time_tracker, LVI_time_tracker,
#                                 LVI_wo_cf_time_tracker, LVI_w_cf_time_tracker)
plot_NSE_bars_with_num_agents(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker, NSE_new_gen_with_cf_tracker,
                              num_of_agents_tracker, Grid)

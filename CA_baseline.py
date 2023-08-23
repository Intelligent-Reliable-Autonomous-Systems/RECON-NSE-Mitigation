import warnings
import simple_colors
import display_lib
from timeit import default_timer as timer
import value_iteration
from blame_assignment import Blame as Blame1
from blame_assignment_baseline import BlameBaseline as Blame2
from init_env import Environment, get_total_R_and_NSE_from_path
from init_env import reset_Agents, show_joint_states_and_NSE_values
from display_lib import plot_NSE_against_CA_baseline, plot_NSE_bars_with_num_agents, show_each_agent_plan

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

for ctr in range(0, len(MM)):

    M = MM[ctr]
    num_of_agents = Num_of_agents[ctr]
    goal_deposit = Goal_deposit[ctr]

    print("-------------------[OUR METHOD]-----------------------")
    print("[OUR METHOD]Number of Agents: ", num_of_agents)
    print("[OUR METHOD]Number of Agents to be corrected: ", M)
    print("[OUR METHOD]Goal deposit: ", goal_deposit)

    mode = 'stochastic'  # 'deterministic' or 'stochastic'
    prob = 0.8
    # initialize the environment
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
    blame = Blame1(Agents, Grid)
    blame_training = blame
    blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    blame.get_training_data_wo_cf(Agents, joint_NSE_states)
    blame.get_training_data_with_cf(Agents, joint_NSE_states)
    Agents = reset_Agents(Agents)

    for agent in Agents:
        agent.generalize_Rblame_linearReg()

    # getting R_blame rewards for each agent by re-simulating original policies
    blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
    print("\n--------Agent-wise NSE blames (before mitigation) [OUR METHOD]--------\n" + str(
        blame_before_mitigation) + " = " + simple_colors.red(str(sum(blame_before_mitigation)), ['bold']))
    sorted_indices = sorted(range(len(blame_before_mitigation)),
                            key=lambda a: blame_before_mitigation[a], reverse=True)
    Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
    print("\nAgents to be corrected: ", Agents_for_correction)
    Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

    Agents = reset_Agents(Agents)

    print("\n---- Now doing Lexicographic Value Iteration with R_blame for selected agents [OUR METHOD]----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report (R_blame):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    R_new, NSE_new = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    # display_all_agent_logs(Agents)

    blame_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame mitigation) [OUR METHOD]--------\n" + str(
        blame_after_R_blame) + " = " + simple_colors.magenta(str(sum(blame_after_R_blame)),
                                                             ['bold']))

    Agents = reset_Agents(Agents)

    print(
        "\n---- Now doing Lexicographic Value Iteration with R_blame_gen_wo_cf for selected agents [OUR METHOD]----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_wo_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_wo_cf):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    blame_after_R_blame_gen_wo_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame_gen mitigation) [OUR METHOD]--------\n" + str(
        blame_after_R_blame_gen_wo_cf) + " = " + simple_colors.blue(str(sum(blame_after_R_blame_gen_wo_cf)), ['bold']))
    R_new_gen_wo_cf, NSE_new_gen_wo_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    Agents = reset_Agents(Agents)

    print(
        "\n---- Now doing Lexicographic Value Iteration with R_blame_gen_with_cf for selected agents [OUR METHOD]----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_with_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_with_cf):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    blame_after_R_blame_gen_with_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n-----Agent-wise NSE (with R_blame_gen mitigation) ------\n" + str(
        blame_after_R_blame_gen_with_cf) + " = " + simple_colors.green(str(sum(blame_after_R_blame_gen_with_cf)),
                                                                       ['bold']))
    R_new_gen_with_cf, NSE_new_gen_with_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    if all_have_reached_goal(Agents):
        print("----------------------------------------------")
        print("[OUR METHOD]Total Reward (initial): ", R_old)
        print("[OUR METHOD]Total NSE (initial): ", simple_colors.red(str(NSE_old), ['bold']))
        print("----------------------------------------------")
        print("[OUR METHOD]Total Reward (after R_blame): ", R_new)
        print("[OUR METHOD]Total NSE (after R_blame): ", simple_colors.magenta(str(NSE_new), ['bold']))
        print("----------------------------------------------")
        print("[OUR METHOD]Total Reward (after R_blame_gen_wo_cf): ", R_new_gen_wo_cf)
        print("[OUR METHOD]Total NSE (after R_blame_gen_wo_cf): ", simple_colors.blue(str(NSE_new_gen_wo_cf), ['bold']))
        print("----------------------------------------------")
        print("[OUR METHOD]Total Reward (after R_blame_gen_with_cf): ", R_new_gen_with_cf)
        print("[OUR METHOD]Total NSE (after R_blame_gen_with_cf): ",
              simple_colors.green(str(NSE_new_gen_with_cf), ['bold']))
        print("----------------------------------------------")

    NSE_old_tracker.append(NSE_old)
    NSE_new_tracker.append(NSE_new)
    NSE_new_gen_wo_cf_tracker.append(NSE_new_gen_wo_cf)
    NSE_new_gen_with_cf_tracker.append(NSE_new_gen_with_cf)
    num_of_agents_tracker.append(num_of_agents)

######################################################################
# ########################## CA BASELINE ########################### #
######################################################################

MM = [1, 2, 2, 3, 4, 6, 12]
Num_of_agents = [2, 3, 4, 5, 8, 12, 20]
Goal_deposit = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 4), (4, 8), (8, 12)]

# Tracking NSE values with grids
NSE_old_tracker_baseline = []
NSE_new_tracker_baseline = []
NSE_new_gen_wo_cf_tracker_baseline = []
NSE_new_gen_with_cf_tracker_baseline = []
num_of_agents_tracker = []
DR_time_tracker = []
for ctr in range(0, len(MM)):

    M = MM[ctr]
    num_of_agents = Num_of_agents[ctr]
    goal_deposit = Goal_deposit[ctr]

    print("--------------------[BASELINE]----------------------")
    print("[BASELINE]Number of Agents: ", num_of_agents)
    print("[BASELINE]Number of Agents to be corrected: ", M)
    print("[BASELINE]Goal deposit: ", goal_deposit)

    mode = 'stochastic'  # 'deterministic' or 'stochastic'
    prob = 0.8
    # initialize the environment
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
    blame = Blame2(Agents, Grid)
    blame_training = blame
    blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    blame.get_training_data_wo_cf(Agents, joint_NSE_states)
    blame.get_training_data_with_cf(Agents, joint_NSE_states)
    Agents = reset_Agents(Agents)

    for agent in Agents:
        agent.generalize_Rblame_linearReg()

    # getting R_blame rewards for each agent by re-simulating original policies
    blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
    print("\n--------Agent-wise NSE blames (before mitigation)[BASELINE] --------\n" + str(
        blame_before_mitigation) + " = " + simple_colors.red(str(sum(blame_before_mitigation)), ['bold']))
    sorted_indices = sorted(range(len(blame_before_mitigation)),
                            key=lambda a: blame_before_mitigation[a], reverse=True)
    Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
    print("\nAgents to be corrected: ", Agents_for_correction)
    Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

    Agents = reset_Agents(Agents)

    print("\n---- Now doing Lexicographic Value Iteration with R_blame for selected agents [BASELINE]----\n ")

    blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    DR_start_time = timer()
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame')
    DR_end_time = timer()
    DR_time = round(float(DR_end_time - DR_start_time) / 60.0, 2)
    DR_time_tracker.append(DR_time)
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents, 'NSE Report (R_blame):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    R_new, NSE_new = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    # display_all_agent_logs(Agents)

    blame_after_R_blame = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame mitigation) --------\n" + str(
        blame_after_R_blame) + " = " + simple_colors.magenta(str(sum(blame_after_R_blame)),
                                                             ['bold']))

    Agents = reset_Agents(Agents)

    print("\n---- Now doing Lexicographic Value Iteration with R_blame_gen_wo_cf for selected agents [BASELINE]----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_wo_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_wo_cf):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    blame_after_R_blame_gen_wo_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n--------Agent-wise NSE (with R_blame_gen mitigation) [BASELINE] --------\n" + str(
        blame_after_R_blame_gen_wo_cf) + " = " + simple_colors.blue(str(sum(blame_after_R_blame_gen_wo_cf)), ['bold']))
    R_new_gen_wo_cf, NSE_new_gen_wo_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    Agents = reset_Agents(Agents)

    print(
        "\n---- Now doing Lexicographic Value Iteration with R_blame_gen_with_cf for selected agents [BASELINE]----\n ")
    value_iteration.LVI(Agents, Agents_to_be_corrected, mode='R_blame_gen_with_cf')
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents,
                                                                               'NSE Report (R_blame_gen_with_cf):')

    blame_distribution_stepwise = []
    for js_nse in joint_NSE_states:
        original_NSE = Grid.give_joint_NSE_value(js_nse)
        blame_values = blame.get_blame(original_NSE, js_nse)
        blame_distribution_stepwise.append(blame_values)

    blame_after_R_blame_gen_with_cf = [sum(x) for x in zip(*blame_distribution_stepwise)]
    print("\n-----Agent-wise NSE (with R_blame_gen mitigation) ------\n" + str(
        blame_after_R_blame_gen_with_cf) + " = " + simple_colors.green(str(sum(blame_after_R_blame_gen_with_cf)),
                                                                       ['bold']))
    R_new_gen_with_cf, NSE_new_gen_with_cf = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)

    if all_have_reached_goal(Agents):
        print("----------------------------------------------")
        print("[BASELINE]Total Reward (initial): ", R_old)
        print("[BASELINE]Total NSE (initial): ", simple_colors.red(str(NSE_old), ['bold']))
        print("----------------------------------------------")
        print("[BASELINE]Total Reward (after R_blame): ", R_new)
        print("[BASELINE]Total NSE (after R_blame): ", simple_colors.magenta(str(NSE_new), ['bold']))
        print("----------------------------------------------")
        print("[BASELINE]Total Reward (after R_blame_gen_wo_cf): ", R_new_gen_wo_cf)
        print("[BASELINE]Total NSE (after R_blame_gen_wo_cf): ", simple_colors.blue(str(NSE_new_gen_wo_cf), ['bold']))
        print("----------------------------------------------")
        print("[BASELINE]Total Reward (after R_blame_gen_with_cf): ", R_new_gen_with_cf)
        print("[BASELINE]Total NSE (after R_blame_gen_with_cf): ",
              simple_colors.green(str(NSE_new_gen_with_cf), ['bold']))
        print("----------------------------------------------")

    NSE_old_tracker_baseline.append(NSE_old)
    NSE_new_tracker_baseline.append(NSE_new)
    NSE_new_gen_wo_cf_tracker_baseline.append(NSE_new_gen_wo_cf)
    NSE_new_gen_with_cf_tracker_baseline.append(NSE_new_gen_with_cf)
    num_of_agents_tracker.append(num_of_agents)
print("DR_time_tracker = ", DR_time_tracker)

plot_NSE_against_CA_baseline(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker, NSE_new_gen_with_cf_tracker,
                             NSE_old_tracker_baseline, NSE_new_tracker_baseline, NSE_new_gen_wo_cf_tracker_baseline,
                             NSE_new_gen_with_cf_tracker_baseline, num_of_agents_tracker, Grid)

# CA Baseline Results
# NSE_old_tracker
# [43.92, 50.91, 85.8, 92.46, 164.1, 279.72, 402.96]
# NSE_new_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 270.3, 386.4]
# NSE_new_gen_wo_cf_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 288.83, 419.78]
# NSE_new_gen_with_cf_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 270.3, 386.4]


# TIms for Normal methods (in minutes)
# LVI_time_tracker = [0.2,0.3,0.4,0.5,1.45,2.2,9.0]
# LVI_wo_cf_time_tracker = [0.21,0.32,0.43,0.53,1.5,2.4,9.8]
# LVI_w_cf_time_tracker = [0.22,0.35,0.45,0.55,1.55,2.5,10.3]

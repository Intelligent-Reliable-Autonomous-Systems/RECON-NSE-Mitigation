import math
import warnings

from timeit import default_timer as timer
from display_lib import *
import value_iteration
from blame_assignment import Blame
from blame_assignment_baseline import BlameBaseline
from init_env import Environment, get_total_R_and_NSE_from_path
from init_env import reset_Agents, show_joint_states_and_NSE_values


warnings.filterwarnings('ignore')

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
agents_to_be_corrected = 0.3  # 30% agents will undergo policy update
Num_of_agents = [2, 5]  # , 10, 25, 50, 75, 100]
MM = [math.ceil(i * agents_to_be_corrected) for i in Num_of_agents]
Goal_deposit = [(1, 1), (2, 3)]  # , (5, 5), (10, 15), (25, 25), (35, 40), (55, 55)]

# Tracking NSE values with grids
NSE_naive_tracker = []
NSE_recon_tracker = []
NSE_recon_gen_wo_cf_tracker = []
NSE_recon_gen_with_cf_tracker = []
NSE_dr_tracker = []
num_of_agents_tracker = []

time_recon_tracker = []
time_gen_recon_wo_cf_tracker = []
time_gen_recon_w_cf_tracker = []
time_dr_tracker = []
num_of_grids = 5

for ctr in range(0, len(MM)):
    NSE_naive_sum = 0
    NSE_recon_sum = 0
    NSE_gen_recon_wo_cf_sum = 0
    NSE_gen_recon_w_cf_sum = 0
    NSE_dr_sum = 0

    time_recon_sum = 0
    time_gen_recon_wo_cf_sum = 0
    time_gen_recon_w_cf_sum = 0
    time_dr_sum = 0

    M = MM[ctr]
    num_of_agents = Num_of_agents[ctr]
    goal_deposit = Goal_deposit[ctr]

    print("------------------------------------------")
    print("Number of Agents: ", num_of_agents)
    print("Number of Agents to be corrected: ", M)
    print("Goal deposit: ", goal_deposit)

    mode = 'stochastic'  # 'deterministic' or 'stochastic'
    prob = 0.8

    for i in [str(x) for x in range(0, num_of_grids)]:
        filename = 'grids/test_grid' + str(i) + '.txt'
        print("======= Now in test_grid" + str(i) + ".txt =======")
        Grid = Environment(num_of_agents, goal_deposit, filename, mode, prob)

        Agents = Grid.init_agents_with_initial_policy()

        ###############################################
        # Naive: initialize agents with the initial coordinating policies
        Agents = reset_Agents(Agents)
        joint_NSE_states, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        print('joint_NSE_values', joint_NSE_values)
        R_naive, NSE_naive = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_naive: ', NSE_naive)
        Agents = reset_Agents(Agents)
        NSE_naive_sum += NSE_naive

        ###############################################
        # RECON (basic Rblame)
        blame = Blame(Agents, Grid)
        blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        Agents = reset_Agents(Agents)

        blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
        sorted_indices = sorted(range(len(blame_before_mitigation)), key=lambda a: blame_before_mitigation[a],
                                reverse=True)
        Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
        print("\nAgents to be corrected [RECON]: ", Agents_for_correction)
        Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]
        Agents = reset_Agents(Agents)

        time_recon_s = timer()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame')  # RECON basic mitigation
        time_recon_e = timer()
        time_recon = round((time_recon_e - time_recon_s) / 60.0, 2)  # in minutes
        joint_NSE_states, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_recon, NSE_recon = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_recon: ', NSE_recon)
        Agents = reset_Agents(Agents)
        NSE_recon_sum += NSE_recon
        time_recon_sum += time_recon

        ###############################################
        # Generalized RECON without counterfactual data
        if int(i) == 0:
            print("Saving training data for wo_cf")
            blame.get_training_data_wo_cf(Agents, joint_NSE_states)
            for agent in Agents:
                filename_agent_x = 'training_data/Agent' + agent.label + '_x_wo_cf.txt'
                filename_agent_y = 'training_data/Agent' + agent.label + '_y_wo_cf.txt'
                np.savetxt(filename_agent_x, agent.blame_training_data_x_wo_cf)
                np.savetxt(filename_agent_y, agent.blame_training_data_y_wo_cf)

        else:
            print("Loading training data for wo_cf")
            for agent in Agents:
                filename_agent_x = 'training_data/Agent' + agent.label + '_x_wo_cf.txt'
                filename_agent_y = 'training_data/Agent' + agent.label + '_y_wo_cf.txt'
                agent.blame_training_data_x_wo_cf = np.loadtxt(filename_agent_x, ndmin=2)
                agent.blame_training_data_y_wo_cf = np.loadtxt(filename_agent_y, ndmin=1)

        for agent in Agents_to_be_corrected:
            agent.generalize_Rblame_wo_cf()

        time_gen_recon_wo_cf_s = timer()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_gen_wo_cf')  # Generalized RECON wo cf
        time_gen_recon_wo_cf_e = timer()
        time_gen_recon_wo_cf = round((time_gen_recon_wo_cf_e - time_gen_recon_wo_cf_s) / 60.0, 2)  # in minutes
        joint_NSE_states, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_gen_recon_wo_cf, NSE_gen_recon_wo_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_gen_recon_wo_cf: ', NSE_gen_recon_wo_cf)
        Agents = reset_Agents(Agents)
        NSE_gen_recon_wo_cf_sum += NSE_gen_recon_wo_cf
        time_gen_recon_wo_cf_sum += time_gen_recon_wo_cf

        ###############################################
        # Generalized RECON with counterfactual data
        if int(i) == 0:
            print("Saving training data for with_cf")
            blame.get_training_data_with_cf(Agents, joint_NSE_states)
            for agent in Agents:
                filename_agent_x = 'training_data/Agent' + agent.label + '_x_with_cf.txt'
                filename_agent_y = 'training_data/Agent' + agent.label + '_y_with_cf.txt'
                np.savetxt(filename_agent_x, agent.blame_training_data_x_with_cf)
                np.savetxt(filename_agent_y, agent.blame_training_data_y_with_cf)

        else:
            print("Loading training data for with_cf")
            for agent in Agents:
                filename_agent_x = 'training_data/Agent' + agent.label + '_x_with_cf.txt'
                filename_agent_y = 'training_data/Agent' + agent.label + '_y_with_cf.txt'
                agent.blame_training_data_x_with_cf = np.loadtxt(filename_agent_x, ndmin=2)
                agent.blame_training_data_y_with_cf = np.loadtxt(filename_agent_y, ndmin=1)

        for agent in Agents_to_be_corrected:
            agent.generalize_Rblame_with_cf()

        time_gen_recon_w_cf_s = timer()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_gen_with_cf')  # Generalized RECON with cf
        time_gen_recon_w_cf_e = timer()
        time_gen_recon_w_cf = round((time_gen_recon_w_cf_e - time_gen_recon_w_cf_s) / 60.0, 2)  # in minutes
        joint_NSE_states, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_gen_recon_w_cf, NSE_gen_recon_w_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_gen_recon_w_cf: ', NSE_gen_recon_w_cf)
        Agents = reset_Agents(Agents)
        NSE_gen_recon_w_cf_sum += NSE_gen_recon_w_cf
        time_gen_recon_w_cf_sum += time_gen_recon_w_cf

        ###############################################
        # Difference Reward Baseline (basic R_blame)
        blameDR = BlameBaseline(Agents, Grid)  # referring to baseline blame calculation using Difference Reward
        blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        Agents = reset_Agents(Agents)

        time_dr_s = timer()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_dr')  # Difference Reward baseline mitigation
        time_dr_e = timer()
        time_dr = round((time_dr_e - time_dr_s) / 60.0, 2)  # in minutes
        joint_NSE_states, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_dr, NSE_dr = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_dr: ', NSE_dr)
        Agents = reset_Agents(Agents)
        NSE_dr_sum += NSE_dr
        time_dr_sum += time_dr

        ##############################################################################################
    NSE_naive_tracker.append(NSE_naive_sum / num_of_grids)
    NSE_recon_tracker.append(NSE_recon_sum / num_of_grids)
    NSE_recon_gen_wo_cf_tracker.append(NSE_gen_recon_wo_cf_sum / num_of_grids)
    NSE_recon_gen_with_cf_tracker.append(NSE_gen_recon_w_cf_sum / num_of_grids)
    NSE_dr_tracker.append(NSE_dr_sum / num_of_grids)

    num_of_agents_tracker.append(num_of_agents)

    time_recon_tracker.append(time_recon_sum / num_of_grids)
    time_gen_recon_wo_cf_tracker.append(time_gen_recon_wo_cf_sum / num_of_grids)
    time_gen_recon_w_cf_tracker.append(time_gen_recon_w_cf_sum / num_of_grids)
    time_dr_tracker.append(time_dr_sum / num_of_grids)

# saving to sim_results_folder
np.savetxt('sim_result_data/NSE_naive_tracker.txt', NSE_naive_tracker, fmt='%.1f')
np.savetxt('sim_result_data/NSE_recon_tracker.txt', NSE_recon_tracker, fmt='%.1f')
np.savetxt('sim_result_data/NSE_recon_gen_wo_cf_tracker.txt', NSE_recon_gen_wo_cf_tracker, fmt='%.1f')
np.savetxt('sim_result_data/NSE_recon_gen_with_cf_tracker.txt', NSE_recon_gen_with_cf_tracker, fmt='%.1f')
np.savetxt('sim_result_data/NSE_dr_tracker.txt', NSE_dr_tracker, fmt='%.1f')
np.savetxt('sim_result_data/num_of_agents_tracker.txt', num_of_agents_tracker, fmt='%d')

np.savetxt('sim_result_data/time_recon_tracker.txt', time_recon_tracker, fmt='%.1f')
np.savetxt('sim_result_data/time_gen_recon_wo_cf_tracker.txt', time_gen_recon_wo_cf_tracker, fmt='%.1f')
np.savetxt('sim_result_data/time_gen_recon_w_cf_tracker.txt', time_gen_recon_w_cf_tracker, fmt='%.1f')
np.savetxt('sim_result_data/time_dr_tracker.txt', time_dr_tracker, fmt='%.1f')

# display_lib.time_plot(num_of_agents_tracker, time_tracker)
# display_lib.separated_time_plot(num_of_agents_tracker, initial_policy_time_tracker, LVI_time_tracker,
#                                 LVI_wo_cf_time_tracker, LVI_w_cf_time_tracker)
# plot_NSE_bars_with_num_agents(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker,
#                               NSE_new_gen_with_cf_tracker, num_of_agents_tracker, Grid)


# CA Baseline Results
# NSE_old_tracker
# [43.92, 50.91, 85.8, 92.46, 164.1, 279.72, 402.96]
# NSE_new_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 270.3, 386.4]
# NSE_new_gen_wo_cf_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 288.83, 419.78]
# NSE_new_gen_with_cf_tracker
# [46.36, 55.67, 90.56, 99.45, 173.22, 270.3, 386.4]

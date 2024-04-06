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
agents_to_be_corrected = 0.5  # 20% agents will undergo policy update
Num_of_agents = [10, 20, 50, 75, 100]
MM = [math.ceil(i * agents_to_be_corrected) for i in Num_of_agents]
Goal_deposit = [(5, 5), (10, 10), (25, 25), (35, 40), (50, 50)]
num_of_grids = 5

# Tracking NSE values with grids

num_of_agents_tracker = []

NSE_naive_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_recon_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_gen_recon_wo_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_gen_recon_with_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_dr_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_considerate_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
NSE_considerate2_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)


time_recon_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
time_gen_recon_wo_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
time_gen_recon_w_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
time_dr_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
time_considerate_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
time_considerate2_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)


for ctr in range(0, len(Num_of_agents)):

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

    for i in [int(x) for x in range(0, num_of_grids)]:
        filename = 'grids/Test_grid' + str(i) + '.txt'
        print("======= Environment: Test_grid" + str(i) + ".txt =======")
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
        NSE_naive_tracker[ctr][i] = NSE_naive

        #for agent in Agents:
        #    agent.follow_policy()
        #    print("Agent " + agent.label + ": " + agent.plan)
        #    agent.agent_reset()

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
        # print("At Line 87 with Test_grid", i)
        blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        # print("At Line 89 with Test_grid", i)
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame')  # RECON basic mitigation
        # print("At Line 91 with Test_grid", i)
        time_recon_e = timer()
        time_recon = round((time_recon_e - time_recon_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_recon, NSE_recon = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_recon: ', NSE_recon)
        Agents = reset_Agents(Agents)
        NSE_recon_tracker[ctr][i] = NSE_recon
        time_recon_tracker[ctr][i] = time_recon

        #for agent in Agents:
        #    agent.follow_policy()
        #    print("Agent " + agent.label + ": " + agent.plan)
        #    agent.agent_reset()
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

        time_gen_recon_wo_cf_s = timer()
        for agent in Agents_to_be_corrected:
            agent.generalize_Rblame_wo_cf()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_gen_wo_cf')  # Generalized RECON wo cf
        time_gen_recon_wo_cf_e = timer()
        time_gen_recon_wo_cf = round((time_gen_recon_wo_cf_e - time_gen_recon_wo_cf_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_gen_recon_wo_cf, NSE_gen_recon_wo_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_gen_recon_wo_cf: ', NSE_gen_recon_wo_cf)
        Agents = reset_Agents(Agents)
        NSE_gen_recon_wo_cf_tracker[ctr][i] = NSE_gen_recon_wo_cf
        time_gen_recon_wo_cf_tracker[ctr][i] = time_gen_recon_wo_cf


        #for agent in Agents:
        #    agent.follow_policy()
        #    print("Agent " + agent.label + ": " + agent.plan)
        #    agent.agent_reset()

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

        time_gen_recon_w_cf_s = timer()
        for agent in Agents_to_be_corrected:
            agent.generalize_Rblame_with_cf()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_gen_with_cf')  # Generalized RECON with cf
        time_gen_recon_w_cf_e = timer()
        time_gen_recon_w_cf = round((time_gen_recon_w_cf_e - time_gen_recon_w_cf_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_gen_recon_w_cf, NSE_gen_recon_w_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_gen_recon_w_cf: ', NSE_gen_recon_w_cf)
        Agents = reset_Agents(Agents)
        NSE_gen_recon_with_cf_tracker[ctr][i] = NSE_gen_recon_w_cf
        time_gen_recon_w_cf_tracker[ctr][i] = time_gen_recon_w_cf


        #for agent in Agents:
        #    agent.follow_policy()
        #    print("Agent " + agent.label + ": " + agent.plan)
        #    agent.agent_reset()

        ###############################################
        # Difference Reward Baseline (basic R_blame)
        blameDR = BlameBaseline(Agents, Grid)  # referring to baseline blame calculation using Difference Reward
        blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        Agents = reset_Agents(Agents)

        time_dr_s = timer()
        blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_dr')  # Difference Reward baseline mitigation
        time_dr_e = timer()
        time_dr = round((time_dr_e - time_dr_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_dr, NSE_dr = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_dr: ', NSE_dr)
        Agents = reset_Agents(Agents)
        NSE_dr_tracker[ctr][i] = NSE_dr
        time_dr_tracker[ctr][i] = time_dr

        ###############################################
        # Be Considerate paper by Parand Alizadeh Alamdari
        # Baseline inspired from [Alizadeh Alamdari et al., 2021]
        # Considerate Reward Baseline (R_blame augmented with other R blames of other agents with caring coefficients)

        time_considerate_s = timer()
        blame.compute_considerate_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        # blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)  # Be considerate baseline using DR
        
        Agents = reset_Agents(Agents)

        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_considerate')  # Difference Reward baseline mitigation
        time_considerate_e = timer()
        time_considerate = round((time_considerate_e - time_considerate_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_considerate, NSE_considerate = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_considerate: ', NSE_considerate)
        Agents = reset_Agents(Agents)
        NSE_considerate_tracker[ctr][i] = NSE_considerate
        time_considerate_tracker[ctr][i] = time_considerate
        
        ###############################################
        # Actual true Be Considerate paper by Parand Alizadeh Alamdari
        # Baseline inspired from [Alizadeh Alamdari et al., 2021]
        # Considerate Reward Baseline (R_blame augmented with other R blames of other agents with caring coefficients)

        blame.compute_scalarized_considerate_R_for_all_Agents(Agents, joint_NSE_states)
        # blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)  # Be considerate baseline using DR
        
        Agents = reset_Agents(Agents)

        time_considerate2_s = timer()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_considerate2')  # Difference Reward baseline mitigation
        time_considerate2_e = timer()
        time_considerate2 = round((time_considerate2_e - time_considerate2_s) / 60.0, 2)  # in minutes
        _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
        R_considerate2, NSE_considerate2 = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
        print('NSE_considerate2: ', NSE_considerate2)
        Agents = reset_Agents(Agents)
        NSE_considerate2_tracker[ctr][i] = NSE_considerate2
        time_considerate2_tracker[ctr][i] = time_considerate2
    ############################################### END of all methods in the for loop

    num_of_agents_tracker.append(num_of_agents)

    print("#########################   AVERAGE SUMMARY   ##########################")
    print("Number of Agents: ", num_of_agents)
    print("NSE_naive (avg): ", np.sum(NSE_naive_tracker[ctr][:]) / num_of_grids)
    print("NSE_recon (avg): ", np.sum(NSE_recon_tracker[ctr][:]) / num_of_grids)
    print("NSE_gen_recon_wo_cf (avg): ", np.sum(NSE_gen_recon_wo_cf_tracker[ctr][:]) / num_of_grids)
    print("NSE_gen_recon_with_cf (avg): ", np.sum(NSE_gen_recon_with_cf_tracker[ctr][:]) / num_of_grids)
    print("NSE_dr (avg): ", np.sum(NSE_dr_tracker[ctr][:]) / num_of_grids)
    print("NSE_considerate (avg): ", np.sum(NSE_considerate_tracker[ctr][:]) / num_of_grids)
    print()
    print("time_recon (avg): ", np.sum(time_recon_tracker[ctr][:]) / num_of_grids)
    print("time_gen_recon_wo_cf (avg): ", np.sum(time_gen_recon_wo_cf_tracker[ctr][:]) / num_of_grids)
    print("time_gen_recon_with_cf (avg): ", np.sum(time_gen_recon_w_cf_tracker[ctr][:]) / num_of_grids)
    print("time_dr (avg): ", np.sum(time_dr_tracker[ctr][:]) / num_of_grids)
    print("time_considerate (avg): ", np.sum(time_considerate_tracker[ctr][:]) / num_of_grids)

    print("########################################################################")

    # saving to sim_results_folder after for all 5 grids in a single row; next row means new number of agents
    np.savetxt('Considerate_sim_results/NSE_naive_tracker.txt', NSE_naive_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_recon_tracker.txt', NSE_recon_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_gen_recon_wo_cf_tracker.txt', NSE_gen_recon_wo_cf_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_gen_recon_with_cf_tracker.txt', NSE_gen_recon_with_cf_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_dr_tracker.txt', NSE_dr_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_considerate_tracker.txt', NSE_considerate_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/NSE_considerate2_tracker.txt', NSE_considerate2_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/num_of_agents_tracker.txt', num_of_agents_tracker, fmt='%d')

    np.savetxt('Considerate_sim_results/time_recon_tracker.txt', time_recon_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/time_gen_recon_wo_cf_tracker.txt', time_gen_recon_wo_cf_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/time_gen_recon_w_cf_tracker.txt', time_gen_recon_w_cf_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/time_dr_tracker.txt', time_dr_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/time_considerate_tracker.txt', time_considerate_tracker, fmt='%.1f')
    np.savetxt('Considerate_sim_results/time_considerate2_tracker.txt', time_considerate2_tracker, fmt='%.1f')

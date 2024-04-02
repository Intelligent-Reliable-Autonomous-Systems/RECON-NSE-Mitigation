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

mode = 'stochastic'  # 'deterministic' or 'stochastic'
prob = 0.8

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
agents_to_be_corrected = 0.5  # 50% agents will undergo policy update
Num_of_agents = 5
Goal_deposit = (2, 3)
num_of_grids = 5
ctr = 0
# Tracking NSE values with grids

num_of_agents_tracker = []

NSE_naive_tracker = np.zeros((1, num_of_grids), dtype=float)
NSE_recon_tracker = np.zeros((1, num_of_grids), dtype=float)
NSE_gen_recon_wo_cf_tracker = np.zeros((1, num_of_grids), dtype=float)
NSE_gen_recon_with_cf_tracker = np.zeros((1, num_of_grids), dtype=float)
NSE_dr_tracker = np.zeros((1, num_of_grids), dtype=float)
NSE_considerate_tracker = np.zeros((1, num_of_grids), dtype=float)

time_recon_tracker = np.zeros((1, num_of_grids), dtype=float)
time_gen_recon_wo_cf_tracker = np.zeros((1, num_of_grids), dtype=float)
time_gen_recon_w_cf_tracker = np.zeros((1, num_of_grids), dtype=float)
time_dr_tracker = np.zeros((1, num_of_grids), dtype=float)
time_considerate_tracker = np.zeros((1, num_of_grids), dtype=float)

num_of_agents = Num_of_agents
M = int(math.ceil(num_of_agents * agents_to_be_corrected))
goal_deposit = Goal_deposit

print("------------------------------------------")
print("Number of Agents: ", num_of_agents)
print("Number of Agents to be corrected: ", M)
print("Goal deposit: ", goal_deposit)
print("mode: ", mode)

for i in [x for x in range(0, num_of_grids)]:
    filename = 'grids/Test_grid' + str(i) + '.txt'
    print("======= Now in Test_grid" + str(i) + ".txt =======")
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
    _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_recon, NSE_recon = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
    print('NSE_recon: ', NSE_recon)
    Agents = reset_Agents(Agents)
    NSE_recon_tracker[ctr][i] = NSE_recon
    time_recon_tracker[ctr][i] = time_recon

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
    _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_gen_recon_wo_cf, NSE_gen_recon_wo_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
    print('NSE_gen_recon_wo_cf: ', NSE_gen_recon_wo_cf)
    Agents = reset_Agents(Agents)
    NSE_gen_recon_wo_cf_tracker[ctr][i] = NSE_gen_recon_wo_cf
    time_gen_recon_wo_cf_tracker[ctr][i] = time_gen_recon_wo_cf

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
    _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_gen_recon_w_cf, NSE_gen_recon_w_cf = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
    print('NSE_gen_recon_w_cf: ', NSE_gen_recon_w_cf)
    Agents = reset_Agents(Agents)
    NSE_gen_recon_with_cf_tracker[ctr][i] = NSE_gen_recon_w_cf
    time_gen_recon_w_cf_tracker[ctr][i] = time_gen_recon_w_cf

    ###############################################
    # Difference Reward Baseline (basic R_blame)
    blameDR = BlameBaseline(Agents, Grid)  # referring to baseline blame calculation using Difference Reward
    blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    Agents = reset_Agents(Agents)

    time_dr_s = timer()
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

    blame.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
    # blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)  # Be considerate baseline using DR
    
    Agents = reset_Agents(Agents)

    time_considerate_s = timer()
    value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_considerate')  # Difference Reward baseline mitigation
    time_considerate_e = timer()
    time_considerate = round((time_considerate_e - time_considerate_s) / 60.0, 2)  # in minutes
    _, joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_considerate, NSE_considerate = get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
    print('NSE_considerate: ', NSE_considerate)
    Agents = reset_Agents(Agents)
    NSE_considerate_tracker[ctr][i] = NSE_considerate
    time_considerate_tracker[ctr][i] = time_considerate

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
# plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
#                               NSE_gen_recon_with_cf_tracker, NSE_dr_tracker, num_of_agents_tracker, 'stochastic')
plot_NSE_bar_comparisons_with_std_mean(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                                       NSE_gen_recon_with_cf_tracker, Grid.mode)

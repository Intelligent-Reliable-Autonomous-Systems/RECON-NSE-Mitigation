import math
import warnings
import numpy as np
import value_iteration
from blame_assignment import Blame
from blame_assignment_baseline import BlameBaseline
from init_env import Environment, get_total_R_and_NSE_from_path
from init_env import reset_Agents, show_joint_states_and_NSE_values

warnings.filterwarnings('ignore')

# Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
MM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # fractions of agents
num_of_agents = 20  # total number of agents to be maintained as constant
Num_agents_to_correct = [math.ceil(num_of_agents * i) for i in MM]
goal_deposit = (10, 10)
mode = 'stochastic'
prob = 0.8

num_of_grids = 5
# Tracking NSE values with grids
NSE_Naive = np.zeros((1, num_of_grids))
# R_Naive = np.float(0.0)

NSE_DR = np.zeros((len(MM), num_of_grids), dtype=float)
# R_DR = np.zeros((len(Num_agents_to_correct),num_of_grids), dtype=float)

NSE_RECON = np.zeros((len(MM), num_of_grids), dtype=float)
# R_RECON = np.zeros((len(Num_agents_to_correct),num_of_grids), dtype=float)
for i in [int(x) for x in range(0, num_of_grids)]:
    filename = 'grids/test_grid' + str(i) + '.txt'
    print("======= Now in test_grid" + str(i) + ".txt =======")
    Grid = Environment(num_of_agents, goal_deposit, filename, mode, prob)

    Agents = Grid.init_agents_with_initial_policy()
    # NAIVE POLICY FOR NAIVE NSE
    # Grid = Environment(num_of_agents, goal_deposit, "grids/train_grid.txt", mode, prob)
    # Agents = Grid.init_agents_with_initial_policy()
    joint_NSE_states, path_joint_NSE_values = show_joint_states_and_NSE_values(Grid, Agents)
    R_Naive, NSE_Naive[0][i] = get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values)
    Agents = reset_Agents(Agents)  # Resetting Agents for next algorithm

    for ctr in range(0, len(MM)):
        M = Num_agents_to_correct[ctr]
        print("-----------------------------------------------------")
        print("------------ " + str(M) + "/" + str(num_of_agents) + " Agents to be corrected ------------")
        print("-----------------------------------------------------")

        #####################################
        # Difference Reward Baseline NSE
        # Now this technique involves blame and generalization over counterfactuals
        blameDR = BlameBaseline(Agents, Grid)  # referring to blame calculation using Difference Reward (Baseline)
        blameDR.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        # blameDR.get_training_data_with_cf(Agents, joint_NSE_states)
        Agents = reset_Agents(Agents)

        blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
        sorted_indices = sorted(range(len(blame_before_mitigation)), key=lambda a: blame_before_mitigation[a],
                                reverse=True)
        Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
        print("\nAgents to be corrected [DR]: ", Agents_for_correction)
        Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

        # blameDR.get_training_data_with_cf(Agents_to_be_corrected, joint_NSE_states)
        # for agent in Agents_to_be_corrected:
        #     agent.generalize_Rblame_with_cf()
        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame_dr')  # Diff Reward Baseline Mitigation
        _, joint_NSE_values_DR = show_joint_states_and_NSE_values(Grid, Agents)
        r_dr, nse_dr = get_total_R_and_NSE_from_path(Agents, joint_NSE_values_DR)

        #####################################
        # RECON (Basic) NSE mitigation
        # This technique is just R_blame from blames and then mitigate NSE
        blame_RECON = Blame(Agents, Grid)  # referring to blame calculation using RECON(basic version)
        blame_RECON.compute_R_Blame_for_all_Agents(Agents, joint_NSE_states)
        # blame_RECON.get_training_data_with_cf(Agents, joint_NSE_states)
        Agents = reset_Agents(Agents)

        blame_before_mitigation = Grid.get_blame_reward_by_following_policy(Agents)
        sorted_indices = sorted(range(len(blame_before_mitigation)), key=lambda a: blame_before_mitigation[a],
                                reverse=True)
        Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
        print("\nAgents to be corrected [RECON]: ", Agents_for_correction)
        Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

        value_iteration.LVI(Agents, Agents_to_be_corrected, 'R_blame')  # RECON basic mitigation
        _, joint_NSE_values_RECON = show_joint_states_and_NSE_values(Grid, Agents)
        r_recon, nse_recon = get_total_R_and_NSE_from_path(Agents, joint_NSE_values_RECON)

        # R_DR[ctr] = r_dr
        NSE_DR[ctr][i] = nse_dr
        # R_RECON[ctr] = r_recon
        NSE_RECON[ctr][i] = nse_recon

MMM = np.array([100 * i for i in MM])
np.savetxt('sim_result_data/NSE_Naive.txt', NSE_Naive, fmt='%.1f')
np.savetxt('sim_result_data/NSE_DR.txt', NSE_DR, fmt='%.1f')
np.savetxt('sim_result_data/NSE_RECON.txt', NSE_RECON, fmt='%.1f')
np.savetxt('sim_result_data/agent_percentage_corrected.txt', MMM, fmt='%d')

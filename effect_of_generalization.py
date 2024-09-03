import math
import warnings
import simple_colors
from display import *
from timeit import default_timer as timer
import compute_policy

warnings.filterwarnings('ignore')
def run_generalization_simulation(domain_name, Agent, Environment, MR):
    # Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
    agents_to_be_corrected = 0.5  # 20% agents will undergo policy update
    Num_of_agents = [10, 20, 50, 75, 100]
    MM = [math.ceil(i * agents_to_be_corrected) for i in Num_of_agents]
    Goal_deposit = [(5, 5), (10, 10), (25, 25), (35, 40), (50, 50)]
    num_of_grids = 5

    # Tracking NSE values with grids
    num_of_agents_tracker = []

    R_naive_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    R_recon_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    R_gen_recon_wo_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    R_gen_recon_with_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    R_dr_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    R_considerate_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)

    NSE_naive_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    NSE_recon_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    NSE_gen_recon_wo_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    NSE_gen_recon_with_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    NSE_dr_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    NSE_considerate_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)


    time_recon_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    time_gen_recon_wo_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    time_gen_recon_w_cf_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    time_dr_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)
    time_considerate_tracker = np.zeros((len(Num_of_agents), num_of_grids), dtype=float)


    for ctr in range(0, len(Num_of_agents)):

        M = MM[ctr]
        num_of_agents = Num_of_agents[ctr]
        goal_deposit = Goal_deposit[ctr]

        print("------------------------------------------")
        print("Number of Agents: ", num_of_agents)
        print("Number of Agents to be corrected: ", M)
        print("Goal deposit: ", goal_deposit)

        for i in [int(x) for x in range(0, num_of_grids)]:
            filename = 'grids/'+domain_name+'/test_grid' + str(i) + '.txt'
            print(simple_colors.yellow('\nEnvironment('+str(i+1)+'/'+str(num_of_grids)+'): '+domain_name+'/test_grid'+str(i),['bold','underlined']))
            # initialize the environment
            Grid = Environment(num_of_agents, goal_deposit, filename)
            # initialize the agents
            Agents = [Agent(i, Grid) for i in range(0, num_of_agents)]
            # initialize the metareasoner
            mr = MR(Agents, Grid)
            
            ###############################################
            # Naive Policy
            print(simple_colors.red("\nNaive Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            for agent in Agents:
                agent.Pi = compute_policy.NaivePolicy(agent)
            joint_NSE_states, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_naive, NSE_naive = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (naive): ", len(joint_NSE_values))
            # print('NSE_naive: ', NSE_naive)
            # print("R_naive = ", sum(R_naive))
            R_naive_tracker[ctr][i] = sum(R_naive)
            NSE_naive_tracker[ctr][i] = NSE_naive
            
            # After naive policy, we find out what agents to be corrected thoughout this simulation
            blame_before_mitigation = mr.get_total_blame_breakdown(joint_NSE_states)
            print("blame_before_mitigation: ", blame_before_mitigation)
            sorted_indices = sorted(range(len(blame_before_mitigation)), key=lambda a: blame_before_mitigation[a],reverse=True)
            Agents_for_correction = ["Agent " + str(i + 1) for i in sorted_indices[:M]]
            print("\nAgents to be corrected: " + str(Agents_for_correction))
            Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]

            ###############################################
            # RECON 
            print(simple_colors.cyan("\nRecon Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            time_recon_s = timer()
            mr.compute_R_Blame_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.ReconPolicy(agent)
            time_recon_e = timer()
            time_recon = round((time_recon_e - time_recon_s) / 60.0, 2)  # in minutes
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_recon, NSE_recon = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (RECON): ", len(joint_NSE_values))
            # print('NSE_recon: ', NSE_recon)
            R_recon_tracker[ctr][i] = sum(R_recon)
            NSE_recon_tracker[ctr][i] = NSE_recon
            time_recon_tracker[ctr][i] = time_recon

            ###############################################
            # Generalized RECON without counterfactual data
            print(simple_colors.green("\nGen Recon without cf Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            if int(i) == 0:
                print("Saving training data for wo_cf")
                mr.get_training_data_wo_cf(Agents, joint_NSE_states)
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
                agent.Pi = compute_policy.GenRecon_wo_cf_Policy(agent)
            time_gen_recon_wo_cf_e = timer()
            time_gen_recon_wo_cf = round((time_gen_recon_wo_cf_e - time_gen_recon_wo_cf_s) / 60.0, 2)  # in minutes
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_gen_recon_wo_cf, NSE_gen_recon_wo_cf = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (gen_recon_wo_cf): ", len(joint_NSE_values))
            # print('NSE_gen_recon_wo_cf: ', NSE_gen_recon_wo_cf)
            R_gen_recon_wo_cf_tracker[ctr][i] = sum(R_gen_recon_wo_cf)
            NSE_gen_recon_wo_cf_tracker[ctr][i] = NSE_gen_recon_wo_cf
            time_gen_recon_wo_cf_tracker[ctr][i] = time_gen_recon_wo_cf

            ###############################################
            # Generalized RECON with counterfactual data
            print(simple_colors.green("\nGen Recon with cf Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            if int(i) == 0:
                print("Saving training data for with_cf")
                mr.get_training_data_with_cf(Agents, joint_NSE_states)
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
                agent.Pi = compute_policy.GenRecon_w_cf_Policy(agent)
            time_gen_recon_w_cf_e = timer()
            time_gen_recon_w_cf = round((time_gen_recon_w_cf_e - time_gen_recon_w_cf_s) / 60.0, 2)  # in minutes
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_gen_recon_w_cf, NSE_gen_recon_w_cf = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (gen_recon_w_cf): ", len(joint_NSE_values))
            # print('NSE_gen_recon_w_cf: ', NSE_gen_recon_w_cf)
            R_gen_recon_with_cf_tracker[ctr][i] = sum(R_gen_recon_w_cf)
            NSE_gen_recon_with_cf_tracker[ctr][i] = NSE_gen_recon_w_cf
            time_gen_recon_w_cf_tracker[ctr][i] = time_gen_recon_w_cf

            ###############################################
            # Difference Reward Baseline
            print(simple_colors.magenta("\nDifference Reward Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            time_dr_s = timer()
            mr.compute_R_Blame_dr_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.DR_Policy(agent)
            time_dr_e = timer()
            time_dr = round((time_dr_e - time_dr_s) / 60.0, 2)  # in minutes
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_dr, NSE_dr = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (DR): ", len(joint_NSE_values))
            # print('NSE_dr: ', NSE_dr)
            R_dr_tracker[ctr][i] = sum(R_dr)
            NSE_dr_tracker[ctr][i] = NSE_dr
            time_dr_tracker[ctr][i] = time_dr

            ###############################################
            # Be Considerate paper by Parand Alizadeh Alamdari
            # Baseline inspired from [Alizadeh Alamdari et al., 2021]
            # Considerate Reward Baseline (R_blame augmented with other R blames of other agents with caring coefficients)
            
            print(simple_colors.blue("\nConsiderate Reward Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]

            time_considerate_s = timer()
            mr.compute_considerate_reward_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.ConsideratePolicy(agent)
            time_considerate_e = timer()
            time_considerate = round((time_considerate_e - time_considerate_s) / 60.0, 2)  # in minutes
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_considerate, NSE_considerate = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            # print("Num of joint steps (considerate): ", len(joint_NSE_values))
            # print('NSE_considerate: ', NSE_considerate)
            R_considerate_tracker[ctr][i] = sum(R_considerate)
            NSE_considerate_tracker[ctr][i] = NSE_considerate
            time_considerate_tracker[ctr][i] = time_considerate
            
        ############################################### END of all methods in the for loop

        num_of_agents_tracker.append(num_of_agents)

        print("#########################   AVERAGE SUMMARY   ##########################")
        print("Number of Agents: ", num_of_agents)
        print("NSE_naive (avg): ", round(np.sum(NSE_naive_tracker[ctr][:]) / num_of_grids,2))
        print("NSE_recon (avg): ", round(np.sum(NSE_recon_tracker[ctr][:]) / num_of_grids,2))
        print("NSE_gen_recon_wo_cf (avg): ", round(np.sum(NSE_gen_recon_wo_cf_tracker[ctr][:]) / num_of_grids,2))
        print("NSE_gen_recon_with_cf (avg): ", round(np.sum(NSE_gen_recon_with_cf_tracker[ctr][:]) / num_of_grids,2))
        print("NSE_dr (avg): ", round(np.sum(NSE_dr_tracker[ctr][:]) / num_of_grids,2))
        print("NSE_considerate (avg): ", round(np.sum(NSE_considerate_tracker[ctr][:]) / num_of_grids,2))
        print()
        print("time_recon (avg): ", np.sum(time_recon_tracker[ctr][:]) / num_of_grids)
        print("time_gen_recon_wo_cf (avg): ", np.sum(time_gen_recon_wo_cf_tracker[ctr][:]) / num_of_grids)
        print("time_gen_recon_with_cf (avg): ", np.sum(time_gen_recon_w_cf_tracker[ctr][:]) / num_of_grids)
        print("time_dr (avg): ", np.sum(time_dr_tracker[ctr][:]) / num_of_grids)
        print("time_considerate (avg): ", np.sum(time_considerate_tracker[ctr][:]) / num_of_grids)

        print("########################################################################")

        # saving to sim_results_folder after for all 5 grids in a single row; next row means new number of agents
        np.savetxt('sim_results/'+domain_name+'/NSE_naive_tracker.txt', NSE_naive_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/NSE_recon_tracker.txt', NSE_recon_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/NSE_gen_recon_wo_cf_tracker.txt', NSE_gen_recon_wo_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/NSE_gen_recon_with_cf_tracker.txt', NSE_gen_recon_with_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/NSE_dr_tracker.txt', NSE_dr_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/NSE_considerate_tracker.txt', NSE_considerate_tracker, fmt='%.1f')
        
        np.savetxt('sim_results/'+domain_name+'/R_naive_tracker.txt', R_naive_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/R_recon_tracker.txt', R_recon_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/R_gen_recon_wo_cf_tracker.txt', R_gen_recon_wo_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/R_gen_recon_with_cf_tracker.txt', R_gen_recon_with_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/R_dr_tracker.txt', R_dr_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/R_considerate_tracker.txt', R_considerate_tracker, fmt='%.1f')

        np.savetxt('sim_results/'+domain_name+'/time_recon_tracker.txt', time_recon_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/time_gen_recon_wo_cf_tracker.txt', time_gen_recon_wo_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/time_gen_recon_w_cf_tracker.txt', time_gen_recon_w_cf_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/time_dr_tracker.txt', time_dr_tracker, fmt='%.1f')
        np.savetxt('sim_results/'+domain_name+'/time_considerate_tracker.txt', time_considerate_tracker, fmt='%.1f')
        
        np.savetxt('sim_results/'+domain_name+'/num_of_agents_tracker.txt', num_of_agents_tracker, fmt='%d')

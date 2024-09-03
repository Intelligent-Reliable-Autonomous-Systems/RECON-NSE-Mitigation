import math
import warnings
import numpy as np
import simple_colors
import compute_policy
warnings.filterwarnings('ignore')

def run_varying_corrected_agents_sim(domain_name, Agent, Environment, MR):
    # Number of agent to be corrected [example (M = 2)/(out of num_of_agents = 5)]
    MM = [0.10, 0.20, 0.50, 0.75, 1.00]  # fractions of agents
    num_of_agents = 25  # total number of agents to be maintained as constant
    Num_agents_to_correct = [math.ceil(num_of_agents * i) for i in MM]
    goal_deposit = (10, 15)

    num_of_grids = 5
    # Tracking NSE values with grids
    NSE_naive_vary = np.zeros((1, num_of_grids))
    NSE_recon_vary = np.zeros((len(MM), num_of_grids), dtype=float)
    NSE_gen_recon_w_cf_vary = np.zeros((len(MM), num_of_grids), dtype=float)
    NSE_dr_vary = np.zeros((len(MM), num_of_grids), dtype=float)
    NSE_considerate_vary = np.zeros((len(MM), num_of_grids))

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
        NSE_naive_vary[0][i] = NSE_naive
        
        # After naive policy, we find out what agents to be corrected thoughout this simulation
        blame_before_mitigation = mr.get_total_blame_breakdown(joint_NSE_states)
        sorted_indices = sorted(range(len(blame_before_mitigation)), key=lambda a: blame_before_mitigation[a],reverse=True)
        for ctr in range(0, len(MM)):
            M = Num_agents_to_correct[ctr]
            Agents_to_be_corrected = [Agents[i] for i in sorted_indices[:M]]
            print("-----------------------------------------------------")
            print("------------ " + str(M) + "/" + str(num_of_agents) + " Agents to be corrected ------------")
            print("-----------------------------------------------------")

            ###############################################
            # RECON 
            print(simple_colors.cyan("\nRecon Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            mr.compute_R_Blame_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.ReconPolicy(agent)
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_recon, NSE_recon = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            NSE_recon_vary[ctr][i] = NSE_recon
            
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

            for agent in Agents_to_be_corrected:
                agent.generalize_Rblame_with_cf()
                agent.Pi = compute_policy.GenRecon_w_cf_Policy(agent)

            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_gen_recon_w_cf, NSE_gen_recon_w_cf = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            NSE_gen_recon_w_cf_vary[ctr][i] = NSE_gen_recon_w_cf
            
            ###############################################
            # Difference Reward Baseline
            print(simple_colors.magenta("\nDifference Reward Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            mr.compute_R_Blame_dr_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.DR_Policy(agent)
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_dr, NSE_dr = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            NSE_dr_vary[ctr][i] = NSE_dr

            ###############################################
            # Be Considerate paper by Parand Alizadeh Alamdari
            # Baseline inspired from [Alizadeh Alamdari et al., 2021]
            # Considerate Reward Baseline (R_blame augmented with other R blames of other agents with caring coefficients)
            
            print(simple_colors.blue("\nConsiderate Reward Policy",['bold','underlined']))
            Agents = [agent.reset() for agent in Agents]
            mr.compute_considerate_reward_for_all_Agents(Agents_to_be_corrected, joint_NSE_states)
            for agent in Agents_to_be_corrected:
                agent.Pi = compute_policy.ConsideratePolicy(agent)
            _, joint_NSE_values = mr.get_jointstates_and_NSE_list(Agents)
            R_considerate, NSE_considerate = mr.get_total_R_and_NSE_from_path(Agents, joint_NSE_values)
            NSE_considerate_vary[ctr][i] = NSE_considerate
            
    MMM = np.array([int(100 * i) for i in MM])
    # np.savetxt('sim_results/'+domain_name+'/Vary_NSE_Naive.txt', NSE_naive_vary, fmt='%.1f')
    # np.savetxt('sim_results/'+domain_name+'/Vary_NSE_DR.txt', NSE_dr_vary, fmt='%.1f')
    # np.savetxt('sim_results/'+domain_name+'/Vary_NSE_RECON.txt', NSE_recon_vary, fmt='%.1f')
    # np.savetxt('sim_results/'+domain_name+'/Vary_NSE_GEN_RECON_with_cf.txt', NSE_gen_recon_w_cf_vary, fmt='%.1f')
    # np.savetxt('sim_results/'+domain_name+'/Vary_NSE_CONSIDERATE.txt', NSE_considerate_vary, fmt='%.1f')
    # np.savetxt('sim_results/'+domain_name+'/Vary_agent_percentage_corrected.txt', MMM, fmt='%d')
    print(simple_colors.green("\nSimulation results saved in sim_results/"+domain_name+"/",['bold']))
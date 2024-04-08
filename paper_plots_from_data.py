from display_lib import *
from init_env import Environment

# print("Hello!")
# # Figure 1: Agents undergoing policy update; simulated on server for 20 Agents [varying_corrected_agents_plot.py]
# # load variables from textfiles
# NSE_naive = np.loadtxt('sim_result_data/results_julia/NSE_Naive.txt', dtype=float)
# NSE_dr = np.loadtxt('sim_result_data/results_julia/NSE_DR.txt', dtype=float)
# NSE_recon = np.loadtxt('sim_result_data/results_julia/NSE_RECON.txt', dtype=float)
# NSE_gen_recon_with_cf = np.loadtxt('sim_result_data/results_julia/NSE_GEN_RECON_with_cf.txt', dtype=float)
# agents_corrected = np.loadtxt('sim_result_data/results_julia/agent_percentage_corrected.txt', dtype=int)
# plot_NSE_LinePlot_with_corrected_agents(NSE_naive, NSE_dr, NSE_recon, NSE_gen_recon_with_cf, agents_corrected)

# Figure 2: NSE penalty from all algorithms averaged over 5 environments [sim_for_varying_num_agents.py]
# load variables from textfiles
NSE_naive_tracker = np.loadtxt('Considerate_sim_results/NSE_naive_tracker.txt', dtype=float)
NSE_considerate_tracker = np.loadtxt('Considerate_sim_results/NSE_considerate2_tracker.txt', dtype=float)
NSE_recon_tracker = np.loadtxt('Considerate_sim_results/NSE_recon_tracker.txt', dtype=float)
NSE_gen_recon_wo_cf_tracker = np.loadtxt('Considerate_sim_results/NSE_gen_recon_wo_cf_tracker.txt', dtype=float)
NSE_gen_recon_with_cf_tracker = np.loadtxt('Considerate_sim_results/NSE_gen_recon_with_cf_tracker.txt', dtype=float)
NSE_dr_tracker = np.loadtxt('Considerate_sim_results/NSE_dr_tracker.txt', dtype=float)
num_of_agents_tracker = np.loadtxt('Considerate_sim_results/num_of_agents_tracker.txt', dtype=int)

# plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
#                               NSE_gen_recon_with_cf_tracker, NSE_dr_tracker, num_of_agents_tracker)

plot_effect_of_generalization2(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                                NSE_gen_recon_with_cf_tracker, NSE_dr_tracker, NSE_considerate_tracker, num_of_agents_tracker)



# Figure 3: Scalability plot showing algorithm times averaged over 5 environments [sim_for_varying_num_agents.py]
# load variables from textfiles
time_recon_tracker = np.loadtxt('Considerate_sim_results/time_recon_tracker.txt', dtype=float) / 60.0
time_gen_recon_wo_cf_tracker = np.loadtxt('Considerate_sim_results/time_gen_recon_wo_cf_tracker.txt', dtype=float) / 60.0
time_gen_recon_w_cf_tracker = np.loadtxt('Considerate_sim_results/time_gen_recon_w_cf_tracker.txt', dtype=float) / 60.0
time_dr_tracker = np.loadtxt('Considerate_sim_results/time_dr_tracker.txt', dtype=float) / 60.0
time_considerate_tracker = np.loadtxt('Considerate_sim_results/time_considerate2_tracker.txt', dtype=float)/ 60.0
num_of_agents_tracker = np.loadtxt('Considerate_sim_results/num_of_agents_tracker.txt', dtype=int)
plot_time_scalability2(time_recon_tracker, time_gen_recon_wo_cf_tracker, time_gen_recon_w_cf_tracker, time_dr_tracker,
                        time_considerate_tracker, num_of_agents_tracker)

# Print the the mean +- standard deviation of the time for 100 agents for each algorithm in minutes rounded to 2 decimal places
print('Time for 100 agents for each algorithm in minutes:')
print('DR: ', np.round(np.mean(time_dr_tracker[-1, :])/60, 2), '+-', np.round(np.std(time_dr_tracker[-1, :])/60, 2))
print('RECON: ', np.round(np.mean(time_recon_tracker[-1, :])/60, 2), '+-', np.round(np.std(time_recon_tracker[-1, :])/60, 2))
print('GEN_RECON_wo_cf: ', np.round(np.mean(time_gen_recon_wo_cf_tracker[-1, :])/60, 2), '+-', np.round(np.std(time_gen_recon_wo_cf_tracker[-1, :])/60, 2))
print('GEN_RECON_w_cf: ', np.round(np.mean(time_gen_recon_w_cf_tracker[-1, :])/60, 2), '+-', np.round(np.std(time_gen_recon_w_cf_tracker[-1, :])/60, 2))






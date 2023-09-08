from display_lib import *
from init_env import Environment

# Figure 1: Agents undergoing policy update; simulated on server for 20 Agents [varying_corrected_agents_plot.py]
# load variables from textfiles
NSE_naive = np.loadtxt('sim_result_data/NSE_Naive3.txt', dtype=float)
NSE_dr = np.loadtxt('sim_result_data/NSE_DR3.txt', dtype=float)
NSE_recon = np.loadtxt('sim_result_data/NSE_RECON3.txt', dtype=float)
NSE_gen_recon_with_cf = np.loadtxt('sim_result_data/NSE_GEN_RECON_with_cf3.txt', dtype=float)
agents_corrected = np.loadtxt('sim_result_data/agent_percentage_corrected3.txt', dtype=int)
plot_NSE_LinePlot_with_corrected_agents(NSE_naive, NSE_dr, NSE_recon, NSE_gen_recon_with_cf, agents_corrected)

# Figure 2: NSE penalty from all algorithms averaged over 5 environments [sim_for_varying_num_agents.py]
# load variables from textfiles
NSE_naive_tracker = np.loadtxt('sim_result_data/NSE_naive_tracker3.txt', dtype=float)
NSE_recon_tracker = np.loadtxt('sim_result_data/NSE_recon_tracker3.txt', dtype=float)
NSE_gen_recon_wo_cf_tracker = np.loadtxt('sim_result_data/NSE_gen_recon_wo_cf_tracker3.txt', dtype=float)
NSE_gen_recon_with_cf_tracker = np.loadtxt('sim_result_data/NSE_gen_recon_with_cf_tracker3.txt', dtype=float)
NSE_dr_tracker = np.loadtxt('sim_result_data/NSE_dr_tracker3.txt', dtype=float)
num_of_agents_tracker = np.loadtxt('sim_result_data/num_of_agents_tracker3.txt', dtype=int)
plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                              NSE_gen_recon_with_cf_tracker, NSE_dr_tracker, num_of_agents_tracker, 'stochastic')


# Figure 3: Scalability plot showing algorithm times averaged over 5 environments [sim_for_varying_num_agents.py]
# load variables from textfiles
time_recon_tracker = np.loadtxt('sim_result_data/time_recon_tracker3.txt', dtype=float)
time_gen_recon_wo_cf_tracker = np.loadtxt('sim_result_data/time_gen_recon_wo_cf_tracker3.txt', dtype=float)
time_gen_recon_w_cf_tracker = np.loadtxt('sim_result_data/time_gen_recon_w_cf_tracker3.txt', dtype=float)
time_dr_tracker = np.loadtxt('sim_result_data/time_dr_tracker3.txt', dtype=float)
plot_time_scalability(time_recon_tracker, time_gen_recon_wo_cf_tracker, time_gen_recon_w_cf_tracker, time_dr_tracker,
                      num_of_agents_tracker)

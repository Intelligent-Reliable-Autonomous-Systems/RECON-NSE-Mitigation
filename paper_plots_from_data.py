from display_lib import *
from init_env import Environment

# Figure 1: Agents undergoing policy update; simulated on server for 20 Agents [varying_corrected_agents_plot.py]
# load variables from textfiles
NSE_naive = np.loadtxt('sim_result_data/NSE_Naive.txt', dtype=float)
NSE_dr = np.loadtxt('sim_result_data/NSE_DR.txt', dtype=float)
NSE_recon = np.loadtxt('sim_result_data/NSE_RECON.txt', dtype=float)
agents_corrected = np.loadtxt('sim_result_data/agent_percentage_corrected.txt', dtype=int)
NSE_with_varying_corrected_agents(NSE_naive, NSE_dr, NSE_recon, agents_corrected)

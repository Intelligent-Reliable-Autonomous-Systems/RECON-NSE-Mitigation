from display_lib import *
from init_env import Environment

# NSE averaged over 10 Grids for 2 Agents (deterministic)

NSE_old_tracker = [73.2, 48.8, 46.36, 43.92, 21.96, 80.52, 48.8, 34.16, 48.8, 73.2]
NSE_new_tracker = [34.16, 31.72, 39.04, 14.64, 12.2, 19.52, 43.92, 43.92, 31.72, 34.16]
NSE_new_gen_wo_cf_tracker = [24.4, 39.04, 9.76, 7.32, 21.96, 9.76, 51.24, 36.6, 39.04, 24.4]
NSE_new_gen_with_cf_tracker = [14.64, 24.4, 2.44, 2.44, 12.2, 2.44, 36.6, 24.4, 24.4, 14.64]

plot_NSE_bar_comparisons_with_std_mean(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker,
                                       NSE_new_gen_with_cf_tracker, 'deterministic')

# NSE averaged over 10 Grids for 2 Agents (stochastic)

NSE_old_tracker = [43.92, 75.64, 19.52, 29.28, 19.52, 29.28, 87.84, 73.2, 75.64, 43.92]
NSE_new_tracker = [34.16, 29.28, 14.64, 14.64, 39.04, 4.88, 41.48, 41.48, 29.28, 34.16]
NSE_new_gen_wo_cf_tracker = [31.72, 29.28, 2.44, 14.64, 24.4, 2.44, 41.48, 41.48, 29.28, 31.72]
NSE_new_gen_with_cf_tracker = [29.28, 26.84, 2.44, 14.64, 24.4, 2.44, 39.04, 39.04, 26.84, 29.28]

plot_NSE_bar_comparisons_with_std_mean(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker,
                                       NSE_new_gen_with_cf_tracker, 'stochastic')

# NSE Penalty with increasing # agents (stochastic)
Grid = Environment(20, (8, 12), "grids/train_grid.txt", 'deterministic', 1)
NSE_old_tracker = [73.2, 84.85, 143.0, 154.1, 273.5, 466.2, 671.6]
NSE_new_tracker = [36.6, 46.03, 71.51, 80.61, 136.76, 285.33, 319.1]
NSE_new_gen_wo_cf_tracker = [24.4, 36.05, 47.68, 58.78, 91.18, 225.04, 201.6]
NSE_new_gen_with_cf_tracker = [24.4, 26.73, 47.68, 49.9, 91.18, 225.04, 201.6]
NSE_new_gen_wo_cf_tracker_baseline = [73.2, 84.85, 143.0, 154.1, 273.5, 466.2, 671.6]
num_of_agents_tracker = [2, 3, 4, 5, 8, 12, 20]
NSE_old_tracker_baseline = NSE_new_tracker_baseline = NSE_new_gen_with_cf_tracker_baseline = []
plot_NSE_against_CA_baseline(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker, NSE_new_gen_with_cf_tracker,
                             NSE_old_tracker_baseline, NSE_new_tracker_baseline, NSE_new_gen_wo_cf_tracker_baseline,
                             NSE_new_gen_with_cf_tracker_baseline, num_of_agents_tracker, Grid)

# NSE Penalty with increasing # agents (stochastic)
Grid = Environment(20, (8, 12), "grids/train_grid.txt", 'stochastic', 0.8)
NSE_old_tracker = [43.92, 50.91, 85.8, 92.46, 164.1, 279.72, 402.96]
NSE_new_tracker = [31.72, 41.15, 61.97, 71.07, 118.52, 219.43, 285.46]
NSE_new_gen_wo_cf_tracker = [31.72, 38.71, 61.97, 68.63, 118.52, 219.43, 285.46]
NSE_new_gen_with_cf_tracker = [31.72, 38.71, 61.97, 68.63, 118.52, 219.43, 285.46]
NSE_new_gen_wo_cf_tracker_baseline = [46.36, 55.67, 90.56, 99.45, 173.22, 288.83, 419.78]
num_of_agents_tracker = [2, 3, 4, 5, 8, 12, 20]
NSE_old_tracker_baseline = NSE_new_tracker_baseline = NSE_new_gen_with_cf_tracker_baseline = []
plot_NSE_against_CA_baseline(NSE_old_tracker, NSE_new_tracker, NSE_new_gen_wo_cf_tracker, NSE_new_gen_with_cf_tracker,
                             NSE_old_tracker_baseline, NSE_new_tracker_baseline, NSE_new_gen_wo_cf_tracker_baseline,
                             NSE_new_gen_with_cf_tracker_baseline, num_of_agents_tracker, Grid)

# Time tracking for all processes with baseline
Num_agents = [2, 3, 4, 5, 8, 12, 20]
LVI_time_tracker = [0.2, 0.3, 0.4, 0.5, 1.45, 2.2, 9.0]
LVI_wo_cf_time_tracker = [0.21, 0.32, 0.43, 0.53, 1.5, 2.4, 9.8]
LVI_w_cf_time_tracker = [0.22, 0.35, 0.45, 0.55, 1.55, 2.5, 10.3]
DR_time_tracker = [0.05, 0.15, 0.25, 0.35, 1.4, 2.0, 8.8]
separated_time_plot(Num_agents, DR_time_tracker, LVI_time_tracker, LVI_wo_cf_time_tracker, LVI_w_cf_time_tracker)

# Centralized vs Decentralized Time (upto 3 agents)

Num_Agents = [1, 2, 3]
Time_Keeping_Centralized = [0.01, 5.9, 338.7]
Time_Keeping_Decentralized = [0.01, 0.2, 0.3]
fig, ax = plt.subplots()

ax.set_xlabel('Number of Agents')
plt.xticks(Num_Agents, Num_Agents)
ax.set_ylabel('Time (min)')
ax.set_title('Centralized Time vs Decentralized Time\nwith number of agents')

plt.plot(Num_Agents, Time_Keeping_Centralized, color='r', label='Centralized')
plt.plot(Num_Agents, Time_Keeping_Decentralized, color='b', label='Decentralized')

ax.legend()
plt.show()

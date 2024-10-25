import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLOR = mcolors.CSS4_COLORS


def plot_NSE_LinePlot_with_corrected_agents(NSE_naive_tracker, NSE_dr_tracker, NSE_recon_tracker,
                                            NSE_gen_recon_with_cf_tracker, NSE_considerate_tracker, 
                                            agents_corrected, domain_name, save_fig=False):
    '''
    :param NSE_naive_tracker: Array of 10 NSE values for Naive Policy
    :param NSE_dr_tracker: Array of 10 NSE values for Difference Reward
    :param NSE_recon_tracker: Array of 10 NSE values for RECON
    :param NSE_gen_recon_with_cf_tracker: Array of 10 NSE values for Generalized RECON with cf data
    :param NSE_considerate_tracker: Array of 10 NSE values for Considerate Reward
    :param agents_corrected: Array of 10 values for number of agents
    :param domain_name: Name of the domain
    :param save_fig: Boolean to save the figure
    :return: plot (Figure 1)
    '''
    num_experiments = len(agents_corrected)  # this could be obtained from length of any other parameter above as well

    fsize = 12
    color1 = COLOR['red']  # color for Naive NSE
    color2 = COLOR['darkorchid']  # color for DR NSE
    color3 = COLOR['darkorange']  # color for RECON NSE
    color4 = COLOR['seagreen']  # color for Gen RECON with cf NSE
    color5 = COLOR['deepskyblue']  # color for Considerate NSE

    title_str = 'NSE mitigation trend with varying percentage\nof agents undergoing policy update in overcooked domain'

    # Calculate the mean and standard deviation for each row
    NSE_naive_means = np.mean(NSE_naive_tracker)
    NSE_naive_means = np.repeat(NSE_naive_means, num_experiments)
    NSE_dr_means = np.mean(NSE_dr_tracker, axis=1)
    NSE_recon_means = np.mean(NSE_recon_tracker, axis=1)
    NSE_gen_recon_with_cf_means = np.mean(NSE_gen_recon_with_cf_tracker, axis=1)
    NSE_considerate_means = np.mean(NSE_considerate_tracker, axis=1)

    NSE_naive_std = np.std(NSE_naive_tracker)
    NSE_dr_std = np.std(NSE_dr_tracker, axis=1)
    NSE_recon_std = np.std(NSE_recon_tracker, axis=1)
    NSE_gen_recon_with_cf_std = np.std(NSE_gen_recon_with_cf_tracker, axis=1)
    NSE_considerate_std = np.std(NSE_considerate_tracker, axis=1)

    plt.rc('axes', labelsize=fsize)
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.plot(agents_corrected, NSE_naive_means, label='Naive Policy', color=color1)
    plt.plot(agents_corrected, NSE_dr_means, label='Difference Reward', color=color2)
    plt.plot(agents_corrected, NSE_considerate_means, label="Considerate Reward "+r'$(\alpha_1= 0.5, \alpha_2= 0.5)$', color=color5)
    plt.plot(agents_corrected, NSE_recon_means, label='RECON', color=color3)
    plt.plot(agents_corrected, NSE_gen_recon_with_cf_means, label='Generalized RECON with cf data', color=color4)

    # Create the shaded region for the standard deviations using fill_between
    plt.fill_between(agents_corrected, NSE_naive_means - NSE_naive_std, NSE_naive_means + NSE_naive_std, alpha=0.1,
                     color=color1)
    plt.fill_between(agents_corrected, NSE_dr_means - NSE_dr_std, NSE_dr_means + NSE_dr_std, alpha=0.1, color=color2)
    plt.fill_between(agents_corrected, NSE_recon_means - NSE_recon_std, NSE_recon_means + NSE_recon_std, alpha=0.1,
                     color=color3)
    plt.fill_between(agents_corrected, NSE_gen_recon_with_cf_means - NSE_gen_recon_with_cf_std,
                     NSE_gen_recon_with_cf_means + NSE_gen_recon_with_cf_std, alpha=0.1, color=color4)
    plt.fill_between(agents_corrected, NSE_considerate_means - NSE_considerate_std,
                        NSE_considerate_means + NSE_considerate_std, alpha=0.1, color=color5)

    plt.xlabel('Percentage of agents undergoing policy update')
    plt.ylabel('NSE Penalty')
    plt.title(title_str)
    plt.legend(fontsize=fsize)

    x_labels = []
    X = np.arange(10, 101, 10, dtype=int)
    for i in X:
        x_labels.append(str(i) + '%')

    plt.xticks([X[i] for i in range(len(X)) if i%2==1], [x_labels[i] for i in range(len(X)) if i%2==1])
    if save_fig:
        plt.savefig("sim_results/"+domain_name+"/Figures/"+domain_name+"_vary_corrected_agents.png", bbox_inches = 'tight', pad_inches = 0)
    plt.show()


def plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                                  NSE_gen_recon_w_cf_tracker, NSE_dr_tracker, NSE_considerate_tracker,
                                  num_agents_tracker, domain_name, save_fig=False):
    '''
    :param NSE_naive_tracker: Array of 10 NSE values for Naive Policy
    :param NSE_recon_tracker: Array of 10 NSE values for RECON
    :param NSE_gen_recon_wo_cf_tracker: Array of 10 NSE values for Generalized RECON without cf data
    :param NSE_gen_recon_w_cf_tracker: Array of 10 NSE values for Generalized RECON with cf data
    :param NSE_dr_tracker: Array of 10 NSE values for Difference Reward
    :param NSE_considerate_tracker: Array of 10 NSE values for Considerate Reward
    :param num_agents_tracker: Array of 10 values for number of agents
    :param domain_name: Name of the domain
    :param save_fig: Boolean to save the figure
    :return: plot (Figure 2)
    '''
    index = np.arange(len(num_agents_tracker))
    bar_width = 0.12
    fsize = 12
    num_grid = 5
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']
    color5 = COLOR['darkorchid']
    color6 = COLOR['deepskyblue']

    title_str = 'NSE penalty for different mitigation techniques for' +'\n' +r'varying number of agents averaged over 5 instances of '+domain_name+' domain'

    fig, ax = plt.subplots()
    # Calculate the mean and standard deviation for each row

    NSE_naive_means = np.mean(NSE_naive_tracker, axis=1)
    NSE_considerate_means = np.mean(NSE_considerate_tracker, axis=1)
    NSE_dr_means = np.mean(NSE_naive_tracker, axis=1) 
    NSE_recon_means = np.mean(NSE_recon_tracker, axis=1) 
    NSE_gen_recon_wo_cf_means = np.mean(NSE_gen_recon_wo_cf_tracker, axis=1)
    NSE_gen_recon_w_cf_means = np.mean(NSE_gen_recon_w_cf_tracker, axis=1)

    NSE_naive_std = np.std(NSE_naive_tracker, axis=1)
    NSE_dr_std = np.std(NSE_dr_tracker, axis=1)
    NSE_recon_std = np.std(NSE_recon_tracker, axis=1)
    NSE_gen_recon_wo_cf_std = np.std(NSE_gen_recon_wo_cf_tracker, axis=1)
    NSE_gen_recon_w_cf_std = np.std(NSE_gen_recon_w_cf_tracker, axis=1)
    NSE_considerate_std = np.std(NSE_considerate_tracker, axis=1)

    ax.bar(index, NSE_naive_means, bar_width, label="Naive Policy", color=color1,
           yerr=NSE_naive_std, ecolor='black', capsize=3)
    ax.bar(index + bar_width, NSE_dr_means, bar_width, label="Difference Reward", color=color5,
           yerr=NSE_dr_std, ecolor='black', capsize=3)
    ax.bar(index + 2 * bar_width, NSE_considerate_means, bar_width, label="Considerate Reward "+r'$(\alpha_1=\alpha_2=0.5)$',
           color=color6, yerr=NSE_considerate_std, ecolor='black', capsize=3)
    ax.bar(index + 3 * bar_width, NSE_recon_means, bar_width, label="RECON", color=color2,
           yerr=NSE_recon_std, ecolor='black', capsize=3)
    ax.bar(index + 4 * bar_width, NSE_gen_recon_wo_cf_means, bar_width, label="Generalized RECON w/o cf data",
           color=color3, yerr=NSE_gen_recon_wo_cf_std, ecolor='black', capsize=3)
    ax.bar(index + 5 * bar_width, NSE_gen_recon_w_cf_means, bar_width, label="Generalized RECON w/ cf data",
           color=color4, yerr=NSE_gen_recon_w_cf_std, ecolor='black', capsize=3)

    ax.set_xticks(index + 2.5 * bar_width)
    x_labels = []
    for i in num_agents_tracker:
        x_labels.append(str(i))

    yticks = [int(i) for i in ax.get_yticks()]
    ax.set_xticklabels(x_labels, fontsize=fsize)
    ax.set_yticklabels(yticks,fontsize=fsize)
    ax.locator_params(nbins=10, axis='y')
    ax.legend(fontsize=fsize)
    ax.set_xlabel('Number of Agents', fontsize=fsize)
    ax.set_ylabel('NSE Penalty', fontsize=fsize)
    ax.set_title(title_str, fontsize=fsize)
    
    if save_fig:
        plt.savefig("sim_results/"+domain_name+"/Figures/"+domain_name+"_effect_of_generalization.png", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    fig.savefig("sim_results/"+domain_name+"/Figures/"+domain_name+"_effect_of_generalization.png", bbox_inches = 'tight', pad_inches = 0)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, fontsize=fsize)
    ax.axis('off')
    fig.savefig("sim_results/"+domain_name+"/Figures/legend_generalization.png", bbox_inches = 'tight', pad_inches = 0.1)
    plt.show()
    
    
    

def plot_time_scalability(time_recon, time_gen_recon_wo_cf, time_gen_recon_w_cf, time_dr, time_considerate, num_of_agents_tracker, domain_name, save_fig=False):
    '''
    :param time_recon: Array of 5 time values for RECON
    :param time_gen_recon_wo_cf: Array of 5 time values for Generalized RECON without cf data
    :param time_gen_recon_w_cf: Array of 5 time values for Generalized RECON with cf data
    :param time_dr: Array of 5 time values for Difference Reward
    :param time_considerate: Array of 5 time values for Considerate Reward
    :param num_of_agents_tracker: Array of 10 values for number of agents
    :param domain_name: Name of the domain
    :param save_fig: Boolean to save the figure
    :return: plot (Figure 3)
    '''
    title_str = 'Process times for varying number of agents\naveraged over 5 instances of '+domain_name+' domain'
    fsize = 14
    color1 = COLOR['darkorange']
    color2 = COLOR['limegreen']
    color3 = COLOR['seagreen']
    color4 = COLOR['darkorchid']
    color5 = COLOR['deepskyblue']

    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Time (min)')
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    x_ticks = np.arange(0, 101, 10, dtype=int)
    plt.xticks(x_ticks, x_ticks)

    time_recon_means = np.mean(time_recon , axis=1)
    time_gen_recon_wo_cf_means = np.mean(time_gen_recon_wo_cf , axis=1)
    time_gen_recon_w_cf_means = np.mean(time_gen_recon_w_cf , axis=1)
    time_dr_means = np.mean(time_dr , axis=1)
    time_considerate_means = np.mean(time_considerate , axis=1)

    time_recon_std = np.std(time_recon , axis=1)
    time_gen_recon_wo_cf_std = np.std(time_gen_recon_wo_cf , axis=1)
    time_gen_recon_w_cf_std = np.std(time_gen_recon_w_cf , axis=1)
    time_dr_std = np.std(time_dr , axis=1)
    time_considerate_std = np.std(time_considerate , axis=1)

    plt.plot(num_of_agents_tracker, time_considerate_means, color=color5, label='Considerate Reward '+r'$(\alpha_1=\alpha_2=0.5)$')
    plt.plot(num_of_agents_tracker, time_dr_means, color=color4, label='Difference Reward')
    plt.plot(num_of_agents_tracker, time_recon_means, color=color1, label='RECON')
    plt.plot(num_of_agents_tracker, time_gen_recon_wo_cf_means, color=color2, label='Generalized RECON without cf data')
    plt.plot(num_of_agents_tracker, time_gen_recon_w_cf_means, color=color3, label='Generalized RECON with cf data')

    plt.fill_between(num_of_agents_tracker, time_recon_means - time_recon_std, time_recon_means + time_recon_std,alpha=0.2, color=color1)
    plt.fill_between(num_of_agents_tracker, time_gen_recon_wo_cf_means - time_gen_recon_wo_cf_std, time_gen_recon_wo_cf_means + time_gen_recon_wo_cf_std, alpha=0.2, color=color2)
    plt.fill_between(num_of_agents_tracker, time_gen_recon_w_cf_means - time_gen_recon_w_cf_std, time_gen_recon_w_cf_means + time_gen_recon_w_cf_std, alpha=0.2, color=color3)
    plt.fill_between(num_of_agents_tracker, time_dr_means - time_dr_std, time_dr_means + time_dr_std, alpha=0.2, color=color4)
    plt.fill_between(num_of_agents_tracker, time_considerate_means - time_considerate_std, time_considerate_means + time_considerate_std, alpha=0.2, color=color5)

    ax.set_xticklabels(x_ticks, fontsize=fsize)
    y_ticks = [int(i) for i in ax.get_yticks()]
    ax.set_yticklabels(y_ticks,fontsize=fsize)
    ax.locator_params(nbins=10, axis='y')
    ax.set_xlabel('Number of Agents', fontsize=fsize)
    ax.set_ylabel('Time (min)', fontsize=fsize)
    plt.title(title_str, fontsize=fsize)
    plt.legend(fontsize=fsize)
    if save_fig:
        plt.savefig("sim_results/"+domain_name+"/Figures/"+domain_name+"_scalability.png", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import init_env
import simple_colors

# rows = init_env.rows
# columns = init_env.columns
# V_arr = np.zeros((rows, columns))
COLOR = mcolors.CSS4_COLORS


# def disp_colormap(V):
#     for s in V:
#         V_arr[s[0]][s[1]] = V[s]
#     # creating a colormap
#     colormap = sns.color_palette("Greens")
#
#     # creating a heatmap using the colormap
#     ax = sns.heatmap(V_arr, cmap=colormap)
#     plt.show()


def display_grid_layout(Grid, Agents):
    # print('\n  Grid Values:\n')
    print_counter = 0
    for i in range(len(Grid)):
        for j in range(len(Grid[0])):
            agent_label = ''
            agent_load_size = ''
            agent_flag = 0
            for ag in Agents:
                if (i, j) == ag.startLoc:
                    agent_label = ag.label
                    agent_load_size = ag.s[2]
                    agent_flag = 1

            if Grid[i][j] == 'C':
                print('%8s' % str("#"), end=" ")
            elif Grid[i][j] == 'S' and agent_flag:
                print('%8s' % str(str(agent_label) + str([agent_load_size])), end=" ")
            elif Grid[i][j] == 'S':
                print('%8s' % str('_'), end=" ")
            else:
                print('%8s' % str(Grid[i][j]), end=" ")
            print_counter += 1
            if print_counter == Grid.columns:
                print_counter = 0
                print()
                print()
    print('\n')


def display_just_grid(Environment):
    print("Environment:")
    for i in range(len(Environment)):
        for j in range(len(Environment[0])):
            print('%8s' % str(Environment[i][j]), end=" ")
        print()
        print()
    print('\n')


def display_values(Grid, V):
    # print('\n  Grid Values:\n')
    print_counter = 0
    for s in V:
        # print('   ' + str(round(V[s])) + '   ', end=" ")
        print('%7s' % str(round(V[s], 1)), end=" ")
        print_counter += 1
        if print_counter == Grid.columns:
            print_counter = 0
            print()
            print()
    print('\n')


def display_policy(Grid, PI):
    # print('\n  Policy :\n')
    print_counter = 0
    for a in PI:
        # print('   ' + str(PI[a]) + '   ', end=" ")
        print('%7s' % str(PI[a]), end=" ")
        print_counter += 1
        if print_counter == Grid.columns:
            print_counter = 0
            print()
            print()
    print('\n')


def display_agent_log(agent):
    print("Path: ", agent.path)
    print("Reward accumulated by Agent " + str(agent.label) + " starting from " + str(agent.startLoc) + " is: ",
          agent.R)


def display_all_agent_logs(Agents):
    print('______________________________________\nAgent Logs:')
    # displaying individual logs of agents -  path, Reward, actual NSE contribution
    for agent in Agents:
        print("\nAgent " + agent.label + ":")
        display_agent_log(agent)


def show_each_agent_plan(Agents):
    for agent in Agents:
        print("Plan for Agent " + agent.label + ":")
        print(agent.plan[4:])  # starting for 4 to avoid the initial arrow display ' -> '
        print("________________________________________________\n")


def plot_reward_bar_comparisons(R_before_mit, R_after_mit, R_after_mit_gen_wo_cf, R_after_mit_gen_w_cf, Grid):
    num_agents = len(R_after_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.4 / num_agents
    fsize = 10
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    title_str = 'Total reward by each agent across mitigation techniques\nfor (' + str(
        Grid.rows) + ' x ' + str(Grid.columns) + ') grid in ' + Grid.mode + ' mode'

    fig, ax = plt.subplots()
    R_values = ax.bar(index, R_before_mit, bar_width, label="Primary-Objective Policy", color=color1)
    R_values_with_blame = ax.bar(index + bar_width, R_after_mit, bar_width, label="RECON",
                                 color=color2)
    R_values_with_blame_gen_wo_cf = ax.bar(index + 2 * bar_width, R_after_mit_gen_wo_cf, bar_width,
                                           label="Generalized RECON without cf data",
                                           color=color3)
    R_values_with_blame_gen_w_cf = ax.bar(index + 3 * bar_width, R_after_mit_gen_w_cf, bar_width,
                                          label="Generalized RECON with cf data",
                                          color=color4)

    ax.set_xlabel('Agents')
    ax.set_ylabel('R')
    plt.ylim([0, plt.ylim()[1] + 50])
    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    x_labels = []
    for i in range(num_agents):
        x_labels.append("Agent " + str(i + 1))

    for bar in R_values_with_blame:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    for bar in R_values:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color1, fontsize=fsize)
    for bar in R_values_with_blame:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color2, fontsize=fsize)
    for bar in R_values_with_blame_gen_wo_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color3, fontsize=fsize)
    for bar in R_values_with_blame_gen_w_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color4, fontsize=fsize)

    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.show()


def plot_NSE_bar_comparisons(NSE_before_mit, NSE_after_mit, NSE_after_mit_gen_wo_cf, NSE_after_mit_gen_w_cf,
                             num_agents_tracker, Grid):
    num_agents = len(NSE_before_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.4 / num_agents
    fsize = 10
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    title_str = 'Total NSE across mitigation techniques \nfor (' + str(
        Grid.rows) + ' x ' + str(Grid.columns) + ') grid in ' + Grid.mode + ' mode'

    fig, ax = plt.subplots()
    NSE_values = ax.bar(index, list(NSE_before_mit), bar_width, label="Primary-Objective Policy", color=color1)
    NSE_values_with_Rblame = ax.bar(index + bar_width, list(NSE_after_mit), bar_width, label="RECON",
                                    color=color2)
    NSE_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, list(NSE_after_mit_gen_wo_cf), bar_width,
                                              label="Generalized RECON without cf data",
                                              color=color3)
    NSE_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, list(NSE_after_mit_gen_w_cf), bar_width,
                                             label="Generalized RECON with cf data",
                                             color=color4)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE')
    plt.ylim([0, plt.ylim()[1] + 50])
    if min(NSE_after_mit_gen_w_cf) == 0.0:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 15, 'Avoidable NSE', color='black', weight='bold',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    else:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 15, 'Unavoidable NSE', color='black', weight='bold',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    x_labels = []
    for i in num_agents_tracker:
        x_labels.append(str(i) + " Agents ")

    for bar in NSE_values:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color1, fontsize=fsize)
    for bar in NSE_values_with_Rblame:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color2, fontsize=fsize)
    for bar in NSE_values_with_Rblame_gen_wo_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color3, fontsize=fsize)
    for bar in NSE_values_with_Rblame_gen_w_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color4, fontsize=fsize)

    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=fsize)

    plt.show()


def plot_NSE_bars_with_num_agents(NSE_before_mit, NSE_after_mit, NSE_after_mit_gen_wo_cf, NSE_after_mit_gen_w_cf,
                                  num_agents_tracker, Grid):
    num_agents = len(NSE_before_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.9 / num_agents
    fsize = 10
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    title_str = 'NSE mitigation trend across varying #agents \n for (' + str(
        Grid.rows) + ' x ' + str(Grid.columns) + ') grid in ' + Grid.mode + ' mode'

    fig, ax = plt.subplots()

    ax.bar(index, NSE_before_mit, bar_width, label="Primary-Objective Policy", color=color1)
    ax.bar(index + bar_width, NSE_after_mit, bar_width, label="RECON",
           color=color2)
    ax.bar(index + 2 * bar_width, NSE_after_mit_gen_wo_cf, bar_width,
           label="Generalized RECON without cf data",
           color=color3)
    ax.bar(index + 3 * bar_width, NSE_after_mit_gen_w_cf, bar_width,
           label="Generalized RECON with cf data",
           color=color4)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] + 50])

    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    x_labels = []
    for i in num_agents_tracker:
        x_labels.append(str(i))

    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=fsize)

    plt.show()


def plot_NSE_against_CA_baseline(NSE_before_mit, NSE_after_mit, NSE_after_mit_gen_wo_cf, NSE_after_mit_gen_w_cf,
                                 NSE_before_mit_b, NSE_after_mit_b, NSE_after_mit_gen_wo_cf_b, NSE_after_mit_gen_w_cf_b,
                                 num_agents_tracker, Grid):
    num_agents = len(NSE_before_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.9 / num_agents
    fsize = 10
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']
    color1b = COLOR['darkorchid']  # ['lightsalmon']
    color2b = COLOR['darkorchid']  # ['wheat']
    color3b = COLOR['darkorchid']  # ['springgreen']
    color4b = COLOR['darkorchid']  # ['mediumspringgreen']

    title_str = 'NSE mitigation trend across varying #agents \n for (' + str(
        Grid.rows) + ' x ' + str(Grid.columns) + ') grid in ' + Grid.mode + ' mode against baseline'

    fig, ax = plt.subplots()

    ax.bar(index, NSE_before_mit, bar_width, label="Primary-Objective Policy", color=color1)
    ax.bar(index + bar_width, NSE_after_mit, bar_width, label="RECON",
           color=color2)
    ax.bar(index + 2 * bar_width, NSE_after_mit_gen_wo_cf, bar_width,
           label="Generalized RECON without cf data",
           color=color3)
    ax.bar(index + 3 * bar_width, NSE_after_mit_gen_w_cf, bar_width,
           label="Generalized RECON with cf data",
           color=color4)
    ax.bar(index + 4 * bar_width, NSE_after_mit_gen_wo_cf_b, bar_width,
           label="Difference Reward", color=color3b)

    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] + 50])

    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    x_labels = []
    for i in num_agents_tracker:
        x_labels.append(str(i))

    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=fsize)

    plt.show()


def plot_NSE_bar_comparisons_with_std_mean(NSE_before_mit_list, NSE_after_mit_list, NSE_after_mit_gen_wo_cf_list,
                                           NSE_after_mit_gen_w_cf_list, mode):
    index = np.arange(1)
    bar_width = 0.4
    fsize = 10
    num_grid = len(NSE_before_mit_list)
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    NSE_before_mean = [np.mean(NSE_before_mit_list)]
    NSE_after_mean = [np.mean(NSE_after_mit_list)]
    NSE_after_gen_mean_wo_cf = [np.mean(NSE_after_mit_gen_wo_cf_list)]
    NSE_after_gen_mean_w_cf = [np.mean(NSE_after_mit_gen_w_cf_list)]

    NSE_before_std = np.std(NSE_before_mit_list)
    NSE_after_std = np.std(NSE_after_mit_list)
    NSE_after_gen_std_wo_cf = np.std(NSE_after_mit_gen_wo_cf_list)
    NSE_after_gen_std_w_cf = np.std(NSE_after_mit_gen_w_cf_list)

    title_str = 'Average total NSE across mitigation techniques\n in ' + str(
        num_grid) + ' similar environments in ' + mode + ' mode'

    fig, ax = plt.subplots()

    NSE_values = ax.bar(index, NSE_before_mean, bar_width, label="Primary-Objective Policy", color=color1,
                        yerr=NSE_before_std, ecolor='black', capsize=10)
    NSE_values_with_Rblame = ax.bar(index + bar_width, NSE_after_mean, bar_width, label="RECON",
                                    color=color2, yerr=NSE_after_std, ecolor='black', capsize=10)
    NSE_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, NSE_after_gen_mean_wo_cf, bar_width,
                                              label="Generalized RECON without cf data",
                                              color=color3, yerr=NSE_after_gen_std_wo_cf, ecolor='black', capsize=10)
    NSE_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, NSE_after_gen_mean_w_cf, bar_width,
                                             label="Generalized RECON with cf data",
                                             color=color4, yerr=NSE_after_gen_std_w_cf, ecolor='black', capsize=10)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] * 2])
    if min(NSE_after_mit_gen_w_cf_list) == 0.0:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 15, 'Avoidable NSE', color='black', weight='bold',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    else:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] * 0.85, '  Unavoidable NSE \nin all environments',
                color='black', weight='bold',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.7'))

    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    x_labels = ["2 Agents"]
    for bar in NSE_values:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + 1.4 * bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color1, fontsize=fsize)
    for bar in NSE_values_with_Rblame:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + 1.4 * bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color2, fontsize=fsize)
    for bar in NSE_values_with_Rblame_gen_wo_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + 1.4 * bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color3, fontsize=fsize)
    for bar in NSE_values_with_Rblame_gen_w_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + 1.4 * bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color4, fontsize=fsize)

    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=fsize)

    plt.show()


def plot_blame_bar_comparisons(blame_before_mit, blame_after_mit, blame_after_mit_gen_wo_cf, blame_after_mit_gen_w_cf,
                               Grid):
    num_agents = len(blame_after_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.4 / num_agents
    fig, ax = plt.subplots()
    fsize = 10
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    title_str = 'Agent-wise blame across mitigation techniques \nfor (' + str(
        Grid.rows) + ' x ' + str(Grid.columns) + ') grid in ' + Grid.mode + ' mode'
    # title_str = 'Agent-wise blame across different mitigation techniques\n' + "(Using pseudo state blames)"

    Blame_values = ax.bar(index, blame_before_mit, bar_width, label="Primary-Objective Policy", color=color1)
    Blame_values_with_Rblame = ax.bar(index + bar_width, blame_after_mit, bar_width, label="RECON",
                                      color=color2)
    Blame_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, blame_after_mit_gen_wo_cf, bar_width,
                                                label="Generalized RECON without cf data", color=color3)
    Blame_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, blame_after_mit_gen_w_cf, bar_width,
                                               label="Generalized RECON with cf data", color=color4)
    ax.set_xlabel('Agents')
    ax.set_ylabel('Blame')
    ax.set_title(title_str)
    ax.set_xticks(index + 1.5 * bar_width)
    plt.ylim([0, plt.ylim()[1] + 50])
    x_labels = []
    for i in range(num_agents):
        x_labels.append("Agent " + str(i + 1))

    for bar in Blame_values:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color1, fontsize=fsize)
    for bar in Blame_values_with_Rblame:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color2, fontsize=fsize)
    for bar in Blame_values_with_Rblame_gen_wo_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color3, fontsize=fsize)
    for bar in Blame_values_with_Rblame_gen_w_cf:
        height = bar.get_height()
        ax.annotate(f'{round(height, 1)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color4, fontsize=fsize)

    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.show()


def time_plot(number_of_agents_tracker, time_tracker):
    title_str = 'Simulation Time vs Number of Agents\n(stochastic transitions)'

    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Agents')
    plt.xticks(number_of_agents_tracker, number_of_agents_tracker)
    ax.set_ylabel('Time (min)')
    # plt.ylim([0, plt.ylim()[1] + 10])
    ax.set_title(title_str)

    plt.plot(number_of_agents_tracker, time_tracker)

    plt.show()


def separated_time_plot(number_of_agents_tracker, DR_time_tracker, LVI_time_tracker, LVI_wo_cf_time_tracker,
                        LVI_w_cf_time_tracker):
    title_str = 'Process Times vs Number of Agents\n(stochastic transitions)'

    color1 = COLOR['darkorchid']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Agents')
    plt.xticks(number_of_agents_tracker, number_of_agents_tracker)
    ax.set_ylabel('Time (min)')

    ax.set_title(title_str)

    plt.plot(number_of_agents_tracker, LVI_time_tracker, color=color2, label='RECON')
    plt.plot(number_of_agents_tracker, LVI_wo_cf_time_tracker, color=color3, label='Generalized RECON without cf data')
    plt.plot(number_of_agents_tracker, LVI_w_cf_time_tracker, color=color4, label='Generalized RECON with cf data')
    plt.plot(number_of_agents_tracker, DR_time_tracker, color=color1, label='Difference Reward')

    ax.legend()
    plt.show()


def plot_NSE_LinePlot_with_corrected_agents(NSE_naive_tracker, NSE_dr_tracker, NSE_recon_tracker,
                                            NSE_gen_recon_with_cf_tracker, agents_corrected):
    """
    :param NSE_naive_tracker: Scalar value of NSE encountered when Agents don't know about NSE
    :param NSE_dr_tracker: Array of 10 NSE values when corrected agents vary from 10% to 100% under Difference Reward baseline
    :param NSE_recon_tracker: Array of 10 NSE values when corrected agents vary from 10% to 100% under RECON (simple R_blame)
    :param NSE_gen_recon_with_cf_tracker: Array of 10 NSE values when corrected agents vary from 10% to 100% under GEN RECON with cf
    :param agents_corrected: Array of 10 NSE values showing percentage of agents corrected

    :return: plot (Figure 1)
    """
    num_experiments = len(agents_corrected)  # this could be obtained from length of any other parameter above as well

    fsize = 10
    color1 = COLOR['red']  # color for Naive NSE
    color2 = COLOR['darkorchid']  # color for DR NSE
    color3 = COLOR['darkorange']  # color for RECON NSE
    color4 = COLOR['seagreen']  # color for Gen RECON with cf NSE

    title_str = 'Average NSE mitigation trend for 25 agents\nwith varying percentage of agents undergoing policy update'

    # Calculate the mean and standard deviation for each row
    NSE_naive_means = np.mean(NSE_naive_tracker)
    NSE_naive_means = np.repeat(NSE_naive_means, num_experiments)
    NSE_dr_means = np.mean(NSE_dr_tracker, axis=1)
    NSE_recon_means = np.mean(NSE_recon_tracker, axis=1)
    NSE_gen_recon_with_cf_means = np.mean(NSE_gen_recon_with_cf_tracker, axis=1)

    NSE_naive_std = np.std(NSE_naive_tracker)
    NSE_dr_std = np.std(NSE_dr_tracker, axis=1)
    NSE_recon_std = np.std(NSE_recon_tracker, axis=1)
    NSE_gen_recon_with_cf_std = np.std(NSE_gen_recon_with_cf_tracker, axis=1)

    # Create the line plot for the means
    plt.plot(agents_corrected, NSE_naive_means, label='Naive Policy', color=color1)
    plt.plot(agents_corrected, NSE_dr_means, label='Difference Reward', color=color2)
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

    plt.plot([50, 50], [0, NSE_gen_recon_with_cf_means[2]], alpha=0.4, linestyle='dotted', color='black')

    plt.xlabel('Percentage of agents undergoing policy update')
    plt.ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] + 50])

    plt.title(title_str)
    # x_ticks = np.array(len(agents_corrected))
    x_labels = []
    X = np.arange(10, 101, 10, dtype=int)
    for i in X:
        x_labels.append(str(i) + '%')

    plt.xticks(X, x_labels)
    plt.legend()
    plt.show()


def plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                                  NSE_gen_recon_w_cf_tracker, NSE_dr_tracker,
                                  num_agents_tracker, mode):
    index = np.arange(len(num_agents_tracker))
    bar_width = 0.15
    fsize = 10
    num_grid = 5
    color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']
    color5 = COLOR['darkorchid']

    title_str = 'Average NSE penalty for different mitigation techniques\n over ' + str(
        num_grid) + ' environments in ' + mode + ' mode'

    fig, ax = plt.subplots()
    # Calculate the mean and standard deviation for each row

    NSE_naive_means = np.mean(NSE_naive_tracker, axis=1)
    NSE_dr_means = np.mean(NSE_dr_tracker, axis=1)
    NSE_recon_means = np.mean(NSE_recon_tracker, axis=1)
    NSE_gen_recon_wo_cf_means = np.mean(NSE_gen_recon_wo_cf_tracker, axis=1)
    NSE_gen_recon_w_cf_means = np.mean(NSE_gen_recon_w_cf_tracker, axis=1)

    NSE_naive_std = np.std(NSE_naive_tracker, axis=1)
    NSE_dr_std = np.std(NSE_dr_tracker, axis=1)
    NSE_recon_std = np.std(NSE_recon_tracker, axis=1)
    NSE_gen_recon_wo_cf_std = np.std(NSE_gen_recon_wo_cf_tracker, axis=1)
    NSE_gen_recon_w_cf_std = np.std(NSE_gen_recon_w_cf_tracker, axis=1)

    ax.bar(index, NSE_naive_means, bar_width, label="Naive Policy", color=color1,
           yerr=NSE_naive_std, ecolor='black', capsize=3)
    ax.bar(index + bar_width, NSE_dr_means, bar_width, label="Difference Reward", color=color5,
           yerr=NSE_dr_std, ecolor='black', capsize=3)
    ax.bar(index + 2 * bar_width, NSE_recon_means, bar_width, label="RECON", color=color2,
           yerr=NSE_recon_std, ecolor='black', capsize=3)
    ax.bar(index + 3 * bar_width, NSE_gen_recon_wo_cf_means, bar_width, label="Generalized RECON without cf data",
           color=color3,
           yerr=NSE_gen_recon_wo_cf_std, ecolor='black', capsize=3)
    ax.bar(index + 4 * bar_width, NSE_gen_recon_w_cf_means, bar_width, label="Generalized RECON with cf data",
           color=color4,
           yerr=NSE_gen_recon_w_cf_std, ecolor='black', capsize=3)

    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] + 25])
    # ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 12, '  Unavoidable NSE \nin all environments',
    #         color='black', weight='bold', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.7'))

    ax.set_title(title_str)
    ax.set_xticks(index + 2.0 * bar_width)
    x_labels = []
    for i in num_agents_tracker:
        x_labels.append(str(i))

    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=fsize)

    plt.show()


def plot_time_scalability(time_recon, time_gen_recon_wo_cf, time_gen_recon_w_cf, time_dr, num_of_agents_tracker):
    title_str = 'Process Times with Number of Agents\nfor (20x20) grids averaged over 5 environment'

    color1 = COLOR['darkorange']
    color2 = COLOR['limegreen']
    color3 = COLOR['seagreen']
    color4 = COLOR['darkorchid']

    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Agents')
    x_ticks = np.arange(0, 101, 10, dtype=int)
    y_ticks = np.arange(0, 11, 1, dtype=int)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    ax.set_ylabel('Time (min)')

    time_recon_means = np.mean(time_recon / 60.0, axis=1)
    time_gen_recon_wo_cf_means = np.mean(time_gen_recon_wo_cf / 60.0, axis=1)
    time_gen_recon_w_cf_means = np.mean(time_gen_recon_w_cf / 60.0, axis=1)
    time_dr_means = np.mean(time_dr / 60.0, axis=1)

    time_recon_std = np.std(time_recon / 60.0, axis=1)
    time_gen_recon_wo_cf_std = np.std(time_gen_recon_wo_cf / 60.0, axis=1)
    time_gen_recon_w_cf_std = np.std(time_gen_recon_w_cf / 60.0, axis=1)
    time_dr_std = np.std(time_dr / 60.0, axis=1)

    ax.set_title(title_str)

    plt.plot(num_of_agents_tracker, time_dr_means, color=color4, label='Difference Reward')
    plt.plot(num_of_agents_tracker, time_recon_means, color=color1, label='RECON')
    plt.plot(num_of_agents_tracker, time_gen_recon_wo_cf_means, color=color2, label='Generalized RECON without cf data')
    plt.plot(num_of_agents_tracker, time_gen_recon_w_cf_means, color=color3, label='Generalized RECON with cf data')

    plt.fill_between(num_of_agents_tracker, time_recon_means - time_recon_std, time_recon_means + time_recon_std,
                     alpha=0.2, color=color1)
    plt.fill_between(num_of_agents_tracker, time_gen_recon_wo_cf_means - time_gen_recon_wo_cf_std,
                     time_gen_recon_wo_cf_means + time_gen_recon_wo_cf_std, alpha=0.2, color=color2)
    plt.fill_between(num_of_agents_tracker, time_gen_recon_w_cf_means - time_gen_recon_w_cf_std,
                     time_gen_recon_w_cf_means + time_gen_recon_w_cf_std, alpha=0.2, color=color3)
    plt.fill_between(num_of_agents_tracker, time_dr_means - time_dr_std, time_dr_means + time_dr_std, alpha=0.2,
                     color=color4)

    ax.legend()
    plt.show()

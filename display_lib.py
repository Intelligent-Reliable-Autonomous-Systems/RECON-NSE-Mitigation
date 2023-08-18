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

    # title_str = 'Reward accumulated by each agent across different mitigation techniques\n' + "(Using pseudo state 
    # blames)"

    fig, ax = plt.subplots()
    R_values = ax.bar(index, R_before_mit, bar_width, label="Initial Policy", color=color1)
    R_values_with_blame = ax.bar(index + bar_width, R_after_mit, bar_width, label="LVI",
                                 color=color2)
    R_values_with_blame_gen_wo_cf = ax.bar(index + 2 * bar_width, R_after_mit_gen_wo_cf, bar_width,
                                           label="Gen LVI without cf data",
                                           color=color3)
    R_values_with_blame_gen_w_cf = ax.bar(index + 3 * bar_width, R_after_mit_gen_w_cf, bar_width,
                                          label="Gen LVI with cf data",
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

    NSE_values = ax.bar(index, NSE_before_mit, bar_width, label="Initial Policy", color=color1)
    NSE_values_with_Rblame = ax.bar(index + bar_width, NSE_after_mit, bar_width, label="LVI",
                                    color=color2)
    NSE_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, NSE_after_mit_gen_wo_cf, bar_width,
                                              label="Gen LVI without cf data",
                                              color=color3)
    NSE_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, NSE_after_mit_gen_w_cf, bar_width,
                                             label="Gen LVI with cf data",
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

    NSE_values = ax.bar(index, NSE_before_mit, bar_width, label="Initial Policy", color=color1)
    NSE_values_with_Rblame = ax.bar(index + bar_width, NSE_after_mit, bar_width, label="LVI",
                                    color=color2)
    NSE_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, NSE_after_mit_gen_wo_cf, bar_width,
                                              label="Gen LVI without cf data",
                                              color=color3)
    NSE_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, NSE_after_mit_gen_w_cf, bar_width,
                                             label="Gen LVI with cf data",
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
    # title_str = 'Average NSE accumulated across 5 similar environments\n' + "(Using pseudo state blames)"

    fig, ax = plt.subplots()

    NSE_values = ax.bar(index, NSE_before_mean, bar_width, label="Initial Policy", color=color1, yerr=NSE_before_std,
                        ecolor='black', capsize=10)
    NSE_values_with_Rblame = ax.bar(index + bar_width, NSE_after_mean, bar_width, label="LVI Mitigation",
                                    color=color2, yerr=NSE_after_std, ecolor='black', capsize=10)
    NSE_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, NSE_after_gen_mean_wo_cf, bar_width,
                                              label="Gen LVI without cf data",
                                              color=color3, yerr=NSE_after_gen_std_wo_cf, ecolor='black', capsize=10)
    NSE_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, NSE_after_gen_mean_w_cf, bar_width,
                                             label="Gen LVI with cf data",
                                             color=color4, yerr=NSE_after_gen_std_w_cf, ecolor='black', capsize=10)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('NSE Penalty')
    plt.ylim([0, plt.ylim()[1] + 20])
    if min(NSE_after_mit_gen_w_cf_list) == 0.0:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 15, 'Avoidable NSE', color='black', weight='bold',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    else:
        ax.text(plt.xlim()[0] + 0.2 * bar_width, plt.ylim()[1] - 12, '  Unavoidable NSE \nin all environments',
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

    Blame_values = ax.bar(index, blame_before_mit, bar_width, label="Initial Policy", color=color1)
    Blame_values_with_Rblame = ax.bar(index + bar_width, blame_after_mit, bar_width, label="LVI ",
                                      color=color2)
    Blame_values_with_Rblame_gen_wo_cf = ax.bar(index + 2 * bar_width, blame_after_mit_gen_wo_cf, bar_width,
                                                label="Gen LVI without cf data", color=color3)
    Blame_values_with_Rblame_gen_w_cf = ax.bar(index + 3 * bar_width, blame_after_mit_gen_w_cf, bar_width,
                                               label="Gen LVI with cf data", color=color4)
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


def separated_time_plot(number_of_agents_tracker, total_experiment_time, LVI_time_tracker, LVI_wo_cf_time_tracker,
                        LVI_w_cf_time_tracker):
    title_str = 'Process Times vs Number of Agents\n(stochastic transitions)'

    # color1 = COLOR['indianred']
    color2 = COLOR['darkorange']
    color3 = COLOR['limegreen']
    color4 = COLOR['seagreen']

    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Agents')
    plt.xticks(number_of_agents_tracker, number_of_agents_tracker)
    ax.set_ylabel('Time (min)')

    ax.set_title(title_str)

    # plt.plot(number_of_agents_tracker, total_experiment_time, color=color1, label='Initial Policy')
    plt.plot(number_of_agents_tracker, LVI_time_tracker, color=color2, label='LVI Mitigation')
    plt.plot(number_of_agents_tracker, LVI_wo_cf_time_tracker, color=color3, label='Gen LVI without cf data')
    plt.plot(number_of_agents_tracker, LVI_w_cf_time_tracker, color=color4, label='Gen LVI with cf data')

    ax.legend()
    plt.show()

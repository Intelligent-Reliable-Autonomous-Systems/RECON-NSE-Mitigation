import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import init_env

rows = init_env.rows
columns = init_env.columns
V_arr = np.zeros((rows, columns))


def disp_colormap(V):
    for s in V:
        V_arr[s[0]][s[1]] = V[s]
    # creating a colormap
    colormap = sns.color_palette("Greens")

    # creating a heatmap using the colormap
    ax = sns.heatmap(V_arr, cmap=colormap)
    plt.show()


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
            if print_counter == columns:
                print_counter = 0
                print()
                print()
    print('\n')


def display_just_grid(Environment):
    for i in range(len(Environment)):
        for j in range(len(Environment[0])):
            print('%8s' % str(Environment[i][j]), end=" ")
        print()
        print()
    print('\n')


def display_values(V):
    # print('\n  Grid Values:\n')
    print_counter = 0
    for s in V:
        # print('   ' + str(round(V[s])) + '   ', end=" ")
        print('%7s' % str(round(V[s], 1)), end=" ")
        print_counter += 1
        if print_counter == columns:
            print_counter = 0
            print()
            print()
    print('\n')


def display_policy(PI):
    # print('\n  Policy :\n')
    print_counter = 0
    for a in PI:
        # print('   ' + str(PI[a]) + '   ', end=" ")
        print('%7s' % str(PI[a]), end=" ")
        print_counter += 1
        if print_counter == columns:
            print_counter = 0
            print()
            print()
    print('\n')


def display_agent_log(agent, flag='before'):
    print("Path: ", agent.path)
    if flag == 'before':
        print("Reward accumulated by Agent " + str(agent.label) + " starting from " + str(agent.startLoc) + " is: ",
              agent.R)
    else:
        print("Reward from R_Blame accumulated by Agent " + str(agent.label) + " starting from " + str(
            agent.startLoc) + " is: ",
              agent.R)
    # print("\n NSE accumulated by Agent " + str(agent.label) + " starting from " + str(agent.startLoc) + " is: ",
    #       agent.NSE)


def plot_reward_bar_comparisons(R_before_mit, R_after_mit, R_after_mit_gen):
    num_agents = len(R_after_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.5 / num_agents

    fig, ax = plt.subplots()
    R_values = ax.bar(index, R_before_mit, bar_width, label="Before Mitigation")
    R_values_with_blame = ax.bar(index + bar_width, R_after_mit, bar_width, label="After Mitigation")
    R_values_with_blame_gen = ax.bar(index + 2 * bar_width, R_after_mit_gen, bar_width, label="After Gen Mitigation")

    ax.set_xlabel('Agents')
    ax.set_ylabel('R')
    plt.ylim(0, 150)
    ax.set_title('Rewards accumulated by each Agent across different modes')
    ax.set_xticks(index + bar_width)
    x_labels = []
    for i in range(num_agents):
        x_labels.append("Agent " + str(i))
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.show()


def plot_NSE_bar_comparisons(NSE_before_mit, NSE_after_mit, NSE_after_mit_gen):
    num_agents = len(NSE_after_mit)  # this could be obtained from length of any other parameter above as well
    index = np.arange(num_agents)
    bar_width = 0.5 / num_agents
    fig, ax = plt.subplots()
    NSE_values = ax.bar(index, NSE_before_mit, bar_width, label="Before Mitigation")
    NSE_values_with_blame = ax.bar(index + bar_width, NSE_after_mit, bar_width, label="After Mitigation")
    NSE_values_with_blame_gen = ax.bar(index + 2 * bar_width, NSE_after_mit_gen, bar_width,
                                       label="After Gen Mitigation")

    ax.set_xlabel('Agents')
    ax.set_ylabel('NSE')
    ax.set_title('NSE from each agent \nacross different mitigation techniques')
    ax.set_xticks(index + bar_width)
    x_labels = []
    for i in range(num_agents):
        x_labels.append("Agent " + str(i))
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.show()

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

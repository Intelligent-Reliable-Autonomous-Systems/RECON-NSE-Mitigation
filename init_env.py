"""
===============================================================================
This script initialized the grid with goal, obstacles and corresponding actions
also other parameters such as gamma

Factored State Representation ->  < x , y , box_with_me , coral_flag , box_at_goal >
    x: denoting x-coordinate of the grid cell
    y: denoting y-coordinate of the grid cell
    box_with_me: junk unit size being carried by the agent {'S','L'}
    coral_flag: boolean denoting presence of coral at the location {True, False}
    box_at_goal: what the goal deposit looks like to this particular agent
===============================================================================
"""
import copy
import read_grid
import numpy as np

goal_deposit = (1, 1)
All_States, rows, columns = read_grid.grid_read_from_file()
all_states = copy.copy(All_States)

# Identifying all goal states G in the All_States grid (list of lists: [[gx_1,gy_1],[gx_2,gy_2]...])
s_goals = np.argwhere(All_States == 'G')
s_goal = s_goals[0]
s_goal = (s_goal[0], s_goal[1], 'X', False, goal_deposit)


class Environment:
    def __init__(self, trash_repository, num_of_agents):
        self.num_of_agents = num_of_agents
        list_of_junk_states = np.argwhere(All_States == 'J')  # this is a list of list [[jx_1,jy_1],[jx_2,jy_2]...]
        self.S, self.coral_flag = initialize_grid_params(trash_repository)
        self.rows = rows
        self.columns = columns
        self.goal_states = s_goals  # this is a list of list [[gx_1,gy_1],[gx_2,gy_2]...]
        self.All_States = All_States
        self.trash_repos = {}
        for [i, j] in list_of_junk_states:
            self.trash_repos[(i, j)] = trash_repository
        self.goal_modes = []
        i = 0
        for i in range(int(s_goal[4][0]) + 1):
            self.goal_modes.append((i, 0))
        for j in range(1, int(s_goal[4][1])):
            self.goal_modes.append((i, j))
        # print('[from init_grid] Goal Modes: ', self.goal_modes)

    def give_joint_NSE_value(self, joint_state):
        # joint_NSE_val = basic_joint_NSE(joint_state)
        # joint_NSE_val = gaussian_joint_NSE(self, joint_state)
        joint_NSE_val = log_joint_NSE(joint_state)
        return joint_NSE_val

    def add_goal_reward(self, agent):
        if agent.s == s_goal:
            agent.R_blame[agent.s] = 100
        return agent

    def max_log_joint_NSE(self, Agents):
        # this function returns worst NSE in the log NSE formulation
        # state s: < x , y , box_with_the_agent , coral_flag(True or False), tuple_of_boxes_at_goal >
        NSE_worst = 10 * np.log(len(Agents) / 20.0 + 1)
        NSE_worst *= 25  # rescaling it to get good values
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst


def move(s, a):
    if a == 'U':
        return s[0] - 1, s[1], s[2], all_states[s[0] - 1][s[1]] == 'C', s[4]
    if a == 'D':
        return s[0] + 1, s[1], s[2], all_states[s[0] + 1][s[1]] == 'C', s[4]
    if a == 'L':
        return s[0], s[1] - 1, s[2], all_states[s[0]][s[1] - 1] == 'C', s[4]
    if a == 'R':
        return s[0], s[1] + 1, s[2], all_states[s[0]][s[1] + 1] == 'C', s[4]
    if a == 'G':
        return s


def do_action(agent, s, a):
    # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    s = list(s)
    if a == 'pick_S':
        s[2] = 'S'
    elif a == 'pick_L':
        s[2] = 'L'
    elif a == 'drop':
        size_index_map = {'S': 0, 'L': 1}
        index = agent.goal_modes.index(s[4])
        # print('[init_env] Agent ' + agent.label + '\'s goal mode was ' + str(s[4]))
        s[4] = list(s[4])
        if (index + agent.num_of_agents) < len(agent.goal_modes):
            s[4] = agent.goal_modes[index + agent.num_of_agents]
        else:
            s[4] = list(s_goal[4])
        s[4] = tuple(s[4])
        s[2] = 'X'
        # print('[init_env] Agent ' + agent.label + '\'s goal mode now is ' + str(s[4]))


    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move(s, a)
    else:
        print("INVALID ACTION (from calc_lib): ", a)
    s = tuple(s)
    return s


# function to find 2d index of any item in the list
# here being used for finding Goal state
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, np.where(x == v)[0][0]


def check_if_in(arr, Arr):
    return any(np.array_equal(x, arr) for x in Arr)


def initialize_grid_params(total_trash_repository):
    # Initializing all states
    # s = < x, y, box_with_me, coral_flag, list_of_boxes_at_goal>
    S = []
    goal_modes = []
    i = 0
    for i in range(int(s_goal[4][0]) + 1):
        goal_modes.append((i, 0))
    for j in range(1, int(s_goal[4][1]) + 1):
        goal_modes.append((i, j))
    for i in range(rows):
        for j in range(columns):
            for box_onboard in ['X', 'L', 'S']:
                for goal_configurations in goal_modes:
                    S.append((i, j, box_onboard, All_States[i][j] == 'C', goal_configurations))

    # recording coral on the floor
    coral = np.zeros((rows, columns), dtype=bool)
    for i in range(rows):
        for j in range(columns):
            if All_States[i][j] == 'C':
                coral[i][j] = True
            else:
                coral[i][j] = False

    # Now go to value_iteration and change Q function
    # and see how and what to pass when calling it and how to pass probability

    return S, coral


def take_step(Grid, Agents):
    NSE_val = 0
    for agent in Agents:
        # print("\n==== Before movement Agent "+agent.label+"'s state: ", agent.s)
        Pi = copy.copy(agent.Pi)
        if agent.s == s_goal:
            # print("Agent "+agent.label+" is at GOAL!")
            continue

        # state of an agent: <x,y,trash_box_size,coral_flag,boxes_at_goal>
        agent.R += agent.Reward(agent.s, Pi[agent.s])
        agent.s = do_action(agent, agent.s, Pi[agent.s])
        # print("\n==== After movement Agent "+agent.label+"'s state: ", agent.s)
        agent.path = agent.path + " -> " + str(agent.s)
    joint_state = get_joint_state(Agents)
    joint_NSE_val = Grid.give_joint_NSE_value(joint_state)

    return Agents, joint_NSE_val


def get_blame_reward_by_following_policy(Agents):
    NSE_blame_dist = []
    for agent in Agents:
        RR = 0
        Pi = copy.copy(agent.Pi)
        while agent.s != s_goal:
            RR += agent.R_blame[agent.s]
            agent.s = do_action(agent, agent.s, Pi[agent.s])
        NSE_blame_dist.append(round(-RR, 2))
    return NSE_blame_dist


def get_joint_state(Agents):
    Joint_State = []
    for agent in Agents:
        Joint_State.append(agent.s)
    return tuple(Joint_State)


def log_joint_NSE(joint_state):
    joint_NSE_val = 0

    weighting = {'X': 0.0, 'S': 3.0, 'L': 10.0}
    X = {'X': 0, 'S': 0, 'L': 0}
    Joint_State = list(copy.deepcopy(joint_state))

    # state s: < x , y , junk_size_being_carried_by_the_agent , coral_flag(True or False) >
    for s in Joint_State:
        if s[3] is True:
            X[s[2]] += 1
    joint_NSE_val = weighting['S'] * np.log(X['S'] / 20.0 + 1) + weighting['L'] * np.log(X['L'] / 20.0 + 1)
    joint_NSE_val *= 25  # rescaling it to get good values
    joint_NSE_val = round(joint_NSE_val, 2)

    return joint_NSE_val

# Grid = Environment({'S': 2, 'L': 3}, 2)
# print(All_States)
# for s in Grid.S:
#     print(s)
# print()
# print("All goal modes: ", Grid.goal_modes)

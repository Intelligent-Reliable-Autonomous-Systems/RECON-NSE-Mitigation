"""
===============================================================================
This script initialized the grid with goal, obstacles and corresponding actions
also other parameters such as gamma

Factored State Representation ->  ( x, y, Traffic_flag)
    x: denoting x-coordinate of the grid cell
    y: denoting y-coordinate of the grid cell
    Traffic_flag: set to 1 if there is traffic in the cell, else 0
===============================================================================
"""
import copy
import read_grid
import numpy as np
from math import exp

# from calculation_lib import move

All_States, rows, columns = read_grid.grid_read_from_file()

all_states = copy.copy(All_States)

# Identifying all goal states G in the All_States grid
s_goals = np.argwhere(All_States == 'G')
# print(s_goals)
zer = np.zeros((len(s_goals), 1))


# s_goals = np.append(s_goals, np.full((len(s_goals), 1), False), axis=1)


class Environment:
    def __init__(self, trash_repository):
        self.S, self.R, self.coral_flag = initialize_grid_params()
        self.rows = rows
        self.columns = columns
        self.trash_repository = trash_repository
        self.goal_states = s_goals  # this is a list of list [[gx_1,gy_1][gx_2,gy_2]...]

    def give_joint_NSE_value(self, joint_state):
        # joint_NSE_val = basic_joint_NSE(joint_state)
        # joint_NSE_val = gaussian_joint_NSE(self, joint_state)
        joint_NSE_val = log_joint_NSE(joint_state)
        return joint_NSE_val

    def add_goal_reward(self, agent):
        for s in s_goals:
            agent.R_blame[tuple(s)] = 100
        return agent

    def max_log_joint_NSE(self, Agents):
        # this function returns worst NSE in the log NSE formulation
        NSE_worst = 0
        X = {'S': 0, 'M': 0, 'L': 0}
        # state s: < x , y , junk_size_being_carried_by_the_agent , coral_flag(True or False) >
        NSE_worst = 10 * np.log(len(Agents) / 20.0 + 1)
        NSE_worst *= 25  # rescaling it to get good values
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst


def move(s, a):
    if a == 'U':
        return s[0] - 1, s[1], s[2], all_states[s[0] - 1][s[1]] == 'C'
    if a == 'D':
        return s[0] + 1, s[1], s[2], all_states[s[0] + 1][s[1]] == 'C'
    if a == 'L':
        return s[0], s[1] - 1, s[2], all_states[s[0]][s[1] - 1] == 'C'
    if a == 'R':
        return s[0], s[1] + 1, s[2], all_states[s[0]][s[1] + 1] == 'C'
    if a == 'G':
        return s


# function to find 2d index of any item in the list
# here being used for finding Goal state
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, np.where(x == v)[0][0]


def check_if_in(arr, Arr):
    return any(np.array_equal(x, arr) for x in Arr)


def initialize_grid_params():
    # Initializing all states
    S = []
    for i in range(rows):
        for j in range(columns):
            if All_States[i][j] == 'C':
                S.append((i, j, True))
            else:
                S.append((i, j, False))

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

    # Defining rewards R for the states
    R = {}
    for s in S:
        # for coral state checking the coral_flag in s[3]
        if [s[0], s[1]] in s_goals:
            R[s] = 100
        else:
            R[s] = -1

    return S, R, coral


def take_step(Grid, Agents):
    NSE_val = 0
    for agent in Agents:
        Pi = copy.copy(agent.Pi)
        if Pi[(agent.s[0], agent.s[1], agent.s[3])] == 'G':
            continue

        # state of an agent: <x,y,trash_size,coral_flag>
        agent.s = move(agent.s, Pi[(agent.s[0], agent.s[1], agent.s[3])])
        agent.R += Grid.R[(agent.s[0], agent.s[1], agent.s[3])]
        agent.path = agent.path + " -> " + str(agent.s)
    joint_state = get_joint_state(Agents)
    joint_NSE_val = Grid.give_joint_NSE_value(joint_state)

    return Agents, joint_NSE_val


def get_reward_by_following_policy(Agents):
    RR_blame_dist = {}
    for agent in Agents:
        Pi = copy.copy(agent.Pi)
        while Pi[(agent.s[0], agent.s[1], agent.s[3])] != 'G':
            agent.s = move(agent.s, Pi[(agent.s[0], agent.s[1], agent.s[3])])
            agent.R += agent.R_blame[(agent.s[0], agent.s[1], agent.s[3])]
        RR_blame_dist[agent.label] = round(agent.R, 2)
    return RR_blame_dist


def get_joint_state(Agents):
    Joint_State = []
    for agent in Agents:
        Joint_State.append(agent.s)
    return tuple(Joint_State)


def log_joint_NSE(joint_state):
    joint_NSE_val = 0
    X = {'S': 0, 'M': 0, 'L': 0}
    Joint_State = list(copy.deepcopy(joint_state))

    # state s: < x , y , junk_size_being_carried_by_the_agent , coral_flag(True or False) >
    for s in Joint_State:
        if s[3] is True:
            X[s[2]] += 1
    joint_NSE_val = 2 * np.log(X['S'] / 20.0 + 1) + 5 * np.log(X['M'] / 20.0 + 1) + 10 * np.log(X['L'] / 20.0 + 1)
    joint_NSE_val *= 25  # rescaling it to get good values
    joint_NSE_val = round(joint_NSE_val, 2)

    return joint_NSE_val

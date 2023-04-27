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
s_goals = np.append(s_goals, np.full((len(s_goals), 1), False), axis=1)


class Environment:
    def __init__(self, trash_repository):
        self.S, self.R, self.NSE, self.Joint_NSE, self.coral_flag = initialize_grid_params()
        self.rows = rows
        self.columns = columns
        self.trash_repository = trash_repository
        self.mu = {'S': 0, 'M': 0, 'L': 0}  # denotes violation threshold in each size category
        self.sigma = {'S': 2, 'M': 5, 'L': 10}  # sensitivity towards each type of size violation
        self.impact_coefficient = {'S': 1, 'M': 2, 'L': 5}
        self.indicator = {'S': False, 'M': False, 'L': False}  # To indicate if violation happened in that category

    def add_jointStateNSE(self, joint_state, NSE_value):
        state_key_list = []
        for s in joint_state:
            key = state_to_hashkey(s)
            state_key_list.append(key)
        jointState_tuple = tuple(state_key_list)
        self.Joint_NSE[jointState_tuple] = NSE_value

    def give_joint_NSE_value(self, joint_state):
        # joint_NSE_val = basic_joint_NSE(joint_state)
        # joint_NSE_val = gaussian_joint_NSE(self, joint_state)
        joint_NSE_val = log_joint_NSE(joint_state)
        return joint_NSE_val

    def add_goal_reward(self, agent):
        for s in s_goals:
            agent.R_blame[tuple(s)] = 100
        return agent

    def max_gaussian_joint_NSE(self, Agents):
        NSE_worst = 0
        X = {'S': 0, 'M': 0, 'L': 0}  # stores how many #agents violating each size category
        mu = copy.copy(self.mu)
        sigma = copy.copy(self.sigma)
        num_agents = len(Agents)
        S = self.trash_repository['S']
        M = self.trash_repository['M']
        L = self.trash_repository['L']
        if num_agents <= L:
            NSE_worst = (-(num_agents - mu['L']) * exp(-(num_agents - mu['L']) ** 2 / (sigma['L']) ** 2))
            NSE_worst *= 10
        elif L < num_agents <= L + M:
            NSE_worst = (-(L - mu['L']) * exp(-(L - mu['L']) ** 2 / (sigma['L']) ** 2)) + (
                    -(num_agents - L - mu['M']) * exp(-(num_agents - L - mu['M']) ** 2 / (sigma['M']) ** 2))
            NSE_worst *= 10
        elif L + M < num_agents <= L + M + S:
            NSE_worst = (-(L - mu['L']) * exp(-(L - mu['L']) ** 2 / (sigma['L']) ** 2)) + (
                    -(M - mu['M']) * exp(-(M - mu['M']) ** 2 / (sigma['M']) ** 2)) + (
                                -(num_agents - M - L - mu['S']) * exp(-(num_agents - M - L - mu['S']) ** 2
                                                                      / (sigma['S']) ** 2))
            NSE_worst *= 10

        # making it positive
        NSE_worst = abs(round(NSE_worst, 2))
        return NSE_worst

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
            # if All_States[i][j] == 'G':
            #     s_goal = (i, j)

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
    NSE = {}  # for singular state NSEs (Not multiagent NSEs)
    NSE_j = {}
    for s in S:
        # for traffic state checking the traffic_flag in s[2]
        if s[2] == 1:
            R[s] = -1
            NSE[s] = 0

        elif check_if_in(np.array([s[0], s[1], s[2]]), s_goals):
            # print("Here!")
            R[s] = 100
            NSE[s] = 0
        # for general state
        else:
            R[s] = -1
            NSE[s] = 0

    return S, R, NSE, NSE_j, coral


def take_step(Grid, Agents):
    NSE_val = 0
    for agent in Agents:
        Pi = copy.copy(agent.Pi)
        if Pi[(agent.s[0], agent.s[1], agent.s[3])] == 'G':
            continue
        agent.s = move(agent.s, Pi[(agent.s[0], agent.s[1], agent.s[3])])
        agent.R += Grid.R[(agent.s[0], agent.s[1], agent.s[3])]
        agent.NSE += Grid.NSE[(agent.s[0], agent.s[1], agent.s[3])]
        x, y, _, _ = agent.s
        NSE_val += Grid.NSE[(agent.s[0], agent.s[1], agent.s[3])]
        loc = (x, y)
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


def get_joint_state_as_keys(Agents):
    Joint_State = []
    joint_state_key = []
    for agent in Agents:
        Joint_State.append(agent.s)
    for s in Joint_State:
        key = state_to_hashkey(s)
        joint_state_key.append(key)
    return tuple(joint_state_key)


def state_to_hashkey(state):
    key = state[0] * columns + state[1] + 1
    return key


def basic_joint_NSE(joint_state):
    joint_NSE_val = 0
    Joint_State = copy.deepcopy(joint_state)
    Joint_State = list(Joint_State)
    coral_flag = [s[3] for s in Joint_State]
    junkUnit_size = [s[2] for s in Joint_State]
    if all(coral_flag) is True:
        for js in junkUnit_size:
            if js == 'L':
                joint_NSE_val += -50
            elif js == 'M':
                joint_NSE_val += -25
            elif js == 'S':
                joint_NSE_val += -10
    # making it positive
    joint_NSE_val = abs(joint_NSE_val)
    return joint_NSE_val


def gaussian_joint_NSE(Grid, joint_state):
    joint_NSE_val = 0
    X = {'S': 0, 'M': 0, 'L': 0}  # stores how #agents violating each size category
    mu = copy.copy(Grid.mu)
    sigma = copy.copy(Grid.sigma)
    indicator = copy.copy(Grid.indicator)

    Joint_State = copy.deepcopy(joint_state)
    Joint_State = list(Joint_State)

    # state s: < x , y , junk_size_being_carried_by_the_agent , coral_flag(True or False) >
    for s in Joint_State:
        if s[3] is True:
            X[s[2]] += 1
    print("-----------X = " + str(X))
    for junk_size in X:
        indicator[junk_size] = (X[junk_size] >= mu[junk_size])
    for junk_size in X:
        joint_NSE_val += 10 * (-(X[junk_size] - mu[junk_size]) * exp(
            -(X[junk_size] - mu[junk_size]) ** 2 / (sigma[junk_size]) ** 2)) * indicator[junk_size]
    joint_NSE_val = round(joint_NSE_val, 2)
    # making if positive
    joint_NSE_val = abs(joint_NSE_val)
    return joint_NSE_val


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

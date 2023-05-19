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
import itertools
from math import exp

All_States, rows, columns = read_grid.grid_read_from_file()

all_states = copy.copy(All_States)

# Identifying all goal states G in the All_States grid (list of lists: [[gx_1,gy_1],[gx_2,gy_2]...])
s_goals = np.argwhere(All_States == 'G')
s_goal = s_goals[0]
s_goal = (s_goal[0], s_goal[1], 'X', False, ('L', 'S'))


class Environment:
    def __init__(self, trash_repository):
        num_of_agents = 2
        self.S, self.coral_flag = initialize_grid_params(num_of_agents)
        self.rows = rows
        self.columns = columns
        self.trash_repository = trash_repository
        self.goal_states = s_goals  # this is a list of list [[gx_1,gy_1],[gx_2,gy_2]...]
        self.All_States = All_States

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


def do_action(s, a):
    s = list(s)
    # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    if a == 'pick_S':
        s[2] = 'S'
    elif a == 'pick_L':
        s[2] = 'L'
    elif a == 'drop':
        if s[4] == 'X':
            s[4] = []
        s[4] = list(s[4])
        s[4].append(s[2])
        s[4].sort()
        s[4] = tuple(s[4])
        s[2] = 'X'
    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move(s, a)
    else:
        print("AT STATE " + str(tuple(s)) + ", INVALID ACTION: " + str(a))
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


def initialize_grid_params(num_of_agents):
    # Initializing all states
    # s = < x, y, box_with_me, coral_flag, list_of_boxes_at_goal>
    S = []
    types = ['L', 'S']
    combinations_at_goal = [b for b in itertools.product(types, repeat=num_of_agents)]
    combinations_at_goal = [list(b) for b in combinations_at_goal]
    combinations_at_goal = [sorted(b) for b in combinations_at_goal]
    Boxes_at_goal = list(set(tuple(sorted(sub)) for sub in combinations_at_goal))
    Boxes_at_goal = [tuple(b) for b in Boxes_at_goal]
    Boxes_at_goal.append(tuple(['X']))
    Boxes_at_goal.append(tuple('L'))
    Boxes_at_goal.append(tuple('S'))
    for i in range(rows):
        for j in range(columns):
            for box_onboard in ['X', 'L', 'S']:
                for boxes_at_goal in Boxes_at_goal:
                    # print("So the box at goal variable here is: ", list(boxes_at_goal))
                    if len(boxes_at_goal) == 2:
                        S.append((i, j, 'X', All_States[i][j] == 'C', boxes_at_goal))
                        continue
                    S.append((i, j, box_onboard, All_States[i][j] == 'C', boxes_at_goal))

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
        print("\n=============== Before movement state: ", agent.s)
        Pi = copy.copy(agent.Pi)
        if agent.s == s_goal:
            continue

        # state of an agent: <x,y,trash_box_size,coral_flag,boxes_at_goal>
        agent.R += agent.Reward(agent.s, Pi[agent.s])
        agent.s = do_action(agent.s, Pi[agent.s])
        print("\n=============== After movement state: ", agent.s)
        agent.path = agent.path + " -> " + str(agent.s)
    joint_state = get_joint_state(Agents)
    joint_NSE_val = Grid.give_joint_NSE_value(joint_state)

    return Agents, joint_NSE_val


def get_reward_by_following_policy(Agents):
    RR_blame_dist = {}
    for agent in Agents:
        Pi = copy.copy(agent.Pi)
        while Pi[(agent.s[0], agent.s[1], agent.s[3])] != 'G':
            agent.s = do_action(agent.s, Pi[(agent.s[0], agent.s[1], agent.s[3])])
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
import init_env
import numpy as np


def check_if_in(arr, Arr):
    return any(np.array_equal(x, arr) for x in Arr)


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, np.where(x == v)


rows = init_env.rows
columns = init_env.columns
s_goals = init_env.s_goals
all_states = init_env.All_States


def get_transition_prob(s, a):
    p_success = 0.8
    p_fail = 0.2
    if (move(s, a) == (-1, -1)) or (check_if_in(np.array([s[0], s[1], s[2]]), s_goals)):  # invalid move
        T = {s: 1}  # stay in same space with prob = 1
    else:
        T = {s: p_fail, move(s, a): p_success}  # (same: 0.2, next: 0.8)

    return T


def get_neighbours(s, a):
    if (move(s, a) == (-1, -1, -1)) or (check_if_in(np.array([s[0], s[1], s[2]]), s_goals)):  # invalid move
        return [s]
    else:
        return [s, move(s, a)]


def get_All_neighbours(s):
    A = ['U', 'D', 'L', 'R']
    Neighbours = []
    for a in A:
        if move(s, a) == (-1, -1, -1):
            continue
        else:
            Neighbours.append(move(s, a))
    return Neighbours


def move(s, a):
    if a == 'U':
        return s[0] - 1, s[1], all_states[s[0] - 1][s[1]] == 'C'
    if a == 'D':
        return s[0] + 1, s[1], all_states[s[0] + 1][s[1]] == 'C'
    if a == 'L':
        return s[0], s[1] - 1, all_states[s[0]][s[1] - 1] == 'C'
    if a == 'R':
        return s[0], s[1] + 1, all_states[s[0]][s[1] + 1] == 'C'
    if a == 'G':
        return s


def get_action(from_state, to_state):
    net_vector = tuple(np.subtract(to_state, from_state))[0:2]
    if net_vector == (0, 1):
        return 'R'
    elif net_vector == (0, -1):
        return 'L'
    elif net_vector == (1, 0):
        return 'D'
    elif net_vector == (-1, 0):
        return 'U'


def all_have_reached_goal(Agents):
    still_not_over_flag = True
    for agent in Agents:
        still_not_over_flag *= (agent.Pi[(agent.s[0], agent.s[1], agent.s[3])] == 'G')
    return still_not_over_flag

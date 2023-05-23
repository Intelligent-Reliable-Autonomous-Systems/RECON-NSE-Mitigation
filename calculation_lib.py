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
s_goal = init_env.s_goals[0]
s_goal = (s_goal[0], s_goal[1], 'X', False, ['L', 'S'])
all_states = init_env.All_States


def get_transition_prob(s, a):
    # actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    p_success = 0.8
    p_fail = 0.2
    if s == s_goal:
        T = {s: 1}  # stay at the goal with prob = 1
    else:
        T = {s: p_fail, do_action(s, a): p_success}  # (same: 0.2, next: 0.8)
    return T


def get_neighbours(s, a):
    if s == s_goal:
        return [s]
    else:
        return [s, do_action(s, a)]


def get_All_neighbours(agent, s):
    Neighbours = []
    for a in agent.A[s]:
        Neighbours.append(do_action(s, a))
    return Neighbours


def move(s, a):
    # note that movement cannot change s[4] (boxes_list_at_goal) so we reuse it from pre-move state itself
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
    # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    s = list(s)
    if a == 'pick_S':
        s[2] = 'S'
    elif a == 'pick_L':
        s[2] = 'L'
    elif a == 'drop':
        size_index_map = {'S': 0, 'L': 1}
        s[4] = list(s[4])
        s[4][size_index_map[s[2]]] += 1
        s[4] = tuple(s[4])
        s[2] = 'X'

    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move(s, a)
    else:
        print("INVALID ACTION (from calc_lib): ", a)
    s = tuple(s)
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
    elif from_state[2] != 'X' and to_state[2] == 'X':
        return 'drop'
    elif from_state[2] == 'X' and to_state[2] == 'S':
        return 'pick_S'
    elif from_state[2] == 'X' and to_state[2] == 'L':
        return 'pick_L'


def all_have_reached_goal(Agents):
    still_not_over_flag = True
    for agent in Agents:
        still_not_over_flag *= (agent.s == s_goal)
    return still_not_over_flag

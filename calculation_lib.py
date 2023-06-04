import init_env
import numpy as np
from init_env import s_goal


def check_if_in(arr, Arr):
    return any(np.array_equal(x, arr) for x in Arr)


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, np.where(x == v)


rows = init_env.rows
columns = init_env.columns
all_states = init_env.All_States


def get_transition_prob(agent, s, a):
    # actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    p_success = 1.0
    p_fail = 0.0
    if s == s_goal:
        T = {s: 1}  # stay at the goal with prob = 1
    else:
        T = {s: p_fail, do_action(agent, s, a): p_success}  # (same: 0.2, next: 0.8)
    return T


def get_neighbours(agent, s, a):
    if s == s_goal:
        return [s]
    else:
        return [s, do_action(agent, s, a)]


def get_All_neighbours(agent, s):
    Neighbours = []
    for a in agent.A[s]:
        Neighbours.append(do_action(agent, s, a))
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
        # print("calc_lib.py DEBUG: 'drop' action at state ", s)
        s[4] = list(s[4])
        if (index + agent.num_of_agents) < len(agent.goal_modes):
            s[4] = agent.goal_modes[index + agent.num_of_agents]
        else:
            s[4] = list(init_env.s_goal[4])
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
    mission_over = True
    for agent in Agents:
        mission_over = mission_over and (agent.s == s_goal)
    # if mission_over:
        # print("[calc_lib] Both Agents are at goal!!")
    # else:
        # print("[calc_lib] It's not over yet!!")
        # print('agent.s: ', Agents[0].s)
        # print('s_goal: ', s_goal)
    return mission_over

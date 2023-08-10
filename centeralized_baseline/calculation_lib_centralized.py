import copy
# from init_env_centralized import CentralizedEnvironment
import numpy as np


def get_transition_prob_centralized(Grid, js, ja):
    """
    :param Grid: Object for class Environment 
    :param js: joint state 
    :param ja: joint action to be done in joint state js
    :return: {js_new: 0.8, js: 0.2}
    """
    # actions = ['pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
    js_new = system_do_action(Grid, js, ja)
    T = {js_new: 0.8, js: 0.2}
    return T


def system_do_action(Grid, js, ja):
    # All agents at the joint state js do the joint action ja (for-loop through them index-wise)
    # js = (s1,s2,...,sn,(goal_status))
    #       s = (x,y, A or B or X , coral_flag at (x,y))
    #       (goal_status) = (#A at goal, #B at goal) before joint action ja is executed
    js_new = []
    goal_status = copy.copy(js[-1])  # goal_status is the tuple at the end of the joint state js (see in comments above)
    goal_status = list(goal_status)
    for agent_idx in range(len(js)-1):
        if ja[agent_idx] == 'drop':
            if js[agent_idx][2] == 'A':
                goal_status[0] += 1
            elif js[agent_idx][2] == 'B':
                goal_status[1] += 1
            else:
                print("[calculation_lib_centralized.py(line 34)] Invalid drop action action")
                exit(0)
        js_new.append(do_action(Grid, js[agent_idx], ja[agent_idx]))
    goal_status = tuple(goal_status)
    js_new.append(goal_status)
    js_new = tuple(js_new)
    return js_new


def move_correctly(Grid, s, a):
    # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    s_next = 0
    if a == 'U':
        if s[0] == 0:
            s_next = s
        else:
            s_next = (s[0] - 1, s[1], s[2], Grid.all_states[s[0] - 1][s[1]] == 'C')
    elif a == 'D':
        if s[0] == Grid.rows - 1:
            s_next = s
        else:
            s_next = (s[0] + 1, s[1], s[2], Grid.all_states[s[0] + 1][s[1]] == 'C')
    elif a == 'L':
        if s[1] == 0:
            s_next = s
        else:
            s_next = (s[0], s[1] - 1, s[2], Grid.all_states[s[0]][s[1] - 1] == 'C')
    elif a == 'R':
        if s[1] == Grid.columns - 1:
            s_next = s
        else:
            s_next = (s[0], s[1] + 1, s[2], Grid.all_states[s[0]][s[1] + 1] == 'C')
    return s_next


def do_action(Grid, s, a):
    # operation actions = ['pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
    s = list(s)
    if a == 'pick_A':
        s[2] = 'A'
    elif a == 'pick_B':
        s[2] = 'B'
    elif a == 'drop':
        s[2] = 'X'
    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move_correctly(Grid, s, a)
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
    elif from_state[2] == 'X' and to_state[2] == 'A':
        return 'pick_A'
    elif from_state[2] == 'X' and to_state[2] == 'B':
        return 'pick_B'


def reached_goal(Grid):
    return Grid.js[-1] == Grid.goal_deposit

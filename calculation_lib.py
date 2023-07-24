import copy
import random
import numpy as np


def get_transition_prob(agent, s, a):
    # actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
    p_success = copy.copy(agent.p_success)
    p_fail = 1 - p_success
    action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    if s == agent.s_goal:
        T = {s: 1}  # stay at the goal with prob = 1
    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s_next_correct = move_correctly(agent.Grid, s, a)
        s_next_slide_left = move_correctly(agent.Grid, s, action[(action_key[a] - 1) % 4])
        s_next_slide_right = move_correctly(agent.Grid, s, action[(action_key[a] + 1) % 4])
        if s_next_correct == s_next_slide_left:
            T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_right: round(p_fail / 2, 3)}
        elif s_next_correct == s_next_slide_right:
            T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_left: round(p_fail / 2, 3)}
        else:
            T = {s_next_correct: round(p_success, 3),
                 s_next_slide_left: round(p_fail / 2, 3),
                 s_next_slide_right: round(p_fail / 2, 3)}

        # T = {s: p_fail, move_correctly(agent.Grid, s, a): p_success}
        # print("T: len = " + str(len(T)) + ":   " + str(T))
    else:
        T = {do_action(agent, s, a): 1}  # (same: 0.2, next: 0.8)
    return T


def move(Grid, s, a):
    action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    p = random.uniform(0, 1)
    if p < 1:
        act = a
    else:
        act = random.choice([action[(action_key[a] - 1) % 4], action[(action_key[a] + 1) % 4]])
    return move_correctly(Grid, s, act)


def move_correctly(Grid, s, a):
    # note that movement cannot change s[4] (samples_list_at_goal) so we reuse it from pre-move state itself
    p = random.uniform(0, 1)
    # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    s_next = 0
    if a == 'U':
        if s[0] == 0:
            s_next = s
        else:
            s_next = (s[0] - 1, s[1], s[2], Grid.all_states[s[0] - 1][s[1]] == 'C', s[4])
    elif a == 'D':
        if s[0] == Grid.rows - 1:
            s_next = s
        else:
            s_next = (s[0] + 1, s[1], s[2], Grid.all_states[s[0] + 1][s[1]] == 'C', s[4])
    elif a == 'L':
        if s[1] == 0:
            s_next = s
        else:
            s_next = (s[0], s[1] - 1, s[2], Grid.all_states[s[0]][s[1] - 1] == 'C', s[4])
    elif a == 'R':
        if s[1] == Grid.columns - 1:
            s_next = s
        else:
            s_next = (s[0], s[1] + 1, s[2], Grid.all_states[s[0]][s[1] + 1] == 'C', s[4])
    return s_next


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
            s[4] = list(agent.s_goal[4])
        s[4] = tuple(s[4])
        s[2] = 'X'

    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move(agent.Grid, s, a)
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
        mission_over = mission_over and (agent.s == agent.s_goal)
    return mission_over

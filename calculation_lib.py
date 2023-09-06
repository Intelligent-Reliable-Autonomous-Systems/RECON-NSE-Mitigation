import copy
import random
import numpy as np


# from init_agent import Agent
# from init_env import Environment


def get_transition_prob(agent, s, a):
    # state of an agent: <x, y, onboard_sample, coral_flag, done_flag>
    # operation actions = ['Noop', 'pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
    p_success = copy.copy(agent.p_success)
    p_fail = 1 - p_success
    action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    if s == agent.s_goal or a == 'Noop':
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
    # state of an agent: <x,y,sample_with_agent,coral_flag,done_flag>
    # operation actions = ['Noop','pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
    s = list(s)
    if a == 'pick_A':
        s[2] = 'A'
    elif a == 'pick_B':
        s[2] = 'B'
    elif a == 'drop':
        # print('\nState: ', s)
        # print("--Assigned sample for Agent " + agent.label + " is " + agent.assigned_sample)
        # print("--Onboard sample for Agent " + agent.label + " is " + str(s[2]))
        # print("Onboard_sample == Assigned_sample: ", agent.assigned_sample == s[2])
        if s[2] == agent.assigned_sample and s[4] is False:
            s[2] = 'X'
            s[4] = True
        else:
            print("[from calc_lib] INVALID 'drop' ACTION at : " + str(s) + " by Agent" + agent.label)
    elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
        s = move(agent.Grid, s, a)
    elif a == 'Noop':
        s = s
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


def all_have_reached_goal(Agents):
    mission_over = True
    for agent in Agents:
        mission_over = mission_over and (agent.s == agent.s_goal)
    return mission_over

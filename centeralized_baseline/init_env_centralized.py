"""
===============================================================================
This script initialized the grid with goal, obstacles and corresponding actions
also other parameters such as gamma

Factored State Representation ->  < x , y , sample_with_me , coral_flag , samples_at_goal_condition >
    x: denoting x-coordinate of the grid cell
    y: denoting y-coordinate of the grid cell
    sample_with_me: junk unit size being carried by the agent {'X','A','B'}
    coral_flag: boolean denoting presence of coral at the location {True, False}
    sample_at_goal_condition: what the goal deposit looks like to this particular agent
===============================================================================
"""
import copy
import read_grid
import numpy as np
from calculation_lib_centralized import system_do_action, reached_goal
from value_iteration_centralized import value_iteration
import simple_colors
import itertools


class CentralizedEnvironment:
    def __init__(self, num_of_agents, goal_deposit, grid_filename, mode, p):

        All_States, rows, columns = read_grid.grid_read_from_file(grid_filename)
        self.all_states = copy.copy(All_States)
        self.goal_deposit = goal_deposit

        self.weighting = {'X': 0.0, 'A': 3.0, 'B': 10.0}

        if mode == 'stochastic':
            self.p_success = p
            print(simple_colors.green('mode: STOCHASTIC', ['bold', 'underlined']))
            print(simple_colors.green('p_success = ' + str(self.p_success), ['bold']))
        elif mode == 'deterministic':
            self.p_success = 1.0
            print(simple_colors.green('mode: DETERMINISTIC', ['bold', 'underlined']))
            print(simple_colors.green('p_success = ' + str(self.p_success), ['bold']))
        else:  # defaulting to deterministic
            self.p_success = 1.0
            print(simple_colors.red('UNKNOWN mode: Defaulting to DETERMINISTIC', ['bold', 'underlined']))
            print(simple_colors.red('p_success = ' + str(self.p_success), ['bold']))
            mode = 'deterministic'

        self.mode = mode
        s_goals = np.argwhere(All_States == 'G')
        s_goal = s_goals[0]
        s0 = [(0, 0, 'X', All_States[0][0] == 'C')]
        self.goal_loc = (s_goal[0], s_goal[1])
        self.V = {}
        self.Pi = {}
        self.Reward = 0.0
        self.plan = ""
        self.path = ""
        self.trajectory = []

        self.js0 = list(itertools.product(s0, repeat=num_of_agents))
        self.js0 = self.js0[0] + ((0, 0),)  # initial goal status at the start
        self.js0 = tuple(self.js0)
        self.js = copy.deepcopy(self.js0)

        self.num_of_agents = num_of_agents
        self.S, self.A = initialize_grid_params(All_States, rows, columns, goal_deposit, self.weighting, num_of_agents)
        self.rows = rows
        self.columns = columns
        self.gamma = 0.99

        self.All_States = All_States
        self.goal_modes = []
        i = 0
        for i in range(int(goal_deposit[0]) + 1):
            self.goal_modes.append((i, 0))
        for j in range(1, int(goal_deposit[1] + 1)):
            self.goal_modes.append((i, j))

    def R(self, js, ja):
        js_next = system_do_action(self, js, ja)
        if js_next[-1] == self.goal_deposit:
            return 100
        else:
            return -1


def initialize_grid_params(All_States, rows, columns, goal_deposit, weighting, num_of_agents):
    # Initializing all states
    # s = < x, y, sample_with_me, coral_flag>
    # js = (s1,s2,s3,...,sn,(goal_status))
    # Joint_States = [js1,js2,js3,...]
    S = []
    goal_statuses = list(itertools.product(tuple(range(goal_deposit[0] + 1)), tuple(range(goal_deposit[1] + 1))))
    for i in range(rows):
        for j in range(columns):
            for sample_onboard in weighting.keys():
                S.append((i, j, sample_onboard, All_States[i][j] == 'C'))

    # recording coral on the floor
    coral = np.zeros((rows, columns), dtype=bool)
    for i in range(rows):
        for j in range(columns):
            if All_States[i][j] == 'C':
                coral[i][j] = True
            else:
                coral[i][j] = False

    # Actions for each singular state (not joint state)
    A = {}
    for s in S:
        A[s] = ['pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']  # operation actions
    for s in S:
        if s[2] != 'X':
            if 'pick_A' in A[s]:
                A[s].remove('pick_A')
            if 'pick_B' in A[s]:
                A[s].remove('pick_B')
        if s[2] == 'X':
            if 'drop' in A[s]:
                A[s].remove('drop')
        if All_States[s[0]][s[1]] != 'A' and All_States[s[0]][s[1]] != 'B':
            if 'pick_A' in A[s]:
                A[s].remove('pick_A')
            if 'pick_B' in A[s]:
                A[s].remove('pick_B')
        if All_States[s[0]][s[1]] != 'G':
            if 'drop' in A[s]:
                A[s].remove('drop')
        # if s[4][0] >= s_goal[4][0]:
        #     if 'pick_A' in A[s]:
        #         A[s].remove('pick_A')
        # if s[4][1] >= s_goal[4][1]:
        #     if 'pick_B' in A[s]:
        #         A[s].remove('pick_B')
        if s[2] == 'S' and s[4][0] > goal_deposit[0]:
            if 'drop' in A[s]:
                A[s].remove('drop')
        if s[2] == 'L' and s[4][1] > goal_deposit[1]:
            if 'drop' in A[s]:
                A[s].remove('drop')
        if All_States[s[0]][s[1]] == 'A':
            if 'pick_B' in A[s]:
                A[s].remove('pick_B')
        if All_States[s[0]][s[1]] == 'B':
            if 'pick_A' in A[s]:
                A[s].remove('pick_A')
    for s in S:
        A[s] = tuple(A[s])

    States = list(itertools.product(S, repeat=num_of_agents))  # making the S into joint states
    Joint_States = []
    for s in States:
        for gs in goal_statuses:
            js = s + (gs,)
            Joint_States.append(js)
    # here s is now a joint state since S has been also changed just before loop
    Actions = {}
    for joint_state in Joint_States:
        Actions[joint_state] = get_joint_action_list(joint_state, A)

    # Now we remove the 'drop' action tuples that violate the goal deposit values
    for js in Joint_States:
        JA = copy.deepcopy(Actions[js])
        A_idxs = [i for i in range(len(js) - 1) if js[i][2] == 'A']
        B_idxs = [i for i in range(len(js) - 1) if js[i][2] == 'B']
        if js[-1][0] + len(A_idxs) > goal_deposit[0]:
            for ja in JA:
                A_drops_idx = [i for i in A_idxs if ja[i] == 'drop']
                if js[-1][0] + len(A_drops_idx) > goal_deposit[0] and ja in Actions[js]:
                    Actions[js].remove(ja)

        if js[-1][1] + len(B_idxs) > goal_deposit[1]:
            for ja in JA:
                B_drops_idx = [i for i in B_idxs if ja[i] == 'drop']
                if js[-1][1] + len(B_drops_idx) > goal_deposit[1] and ja in Actions[js]:
                    Actions[js].remove(ja)

    return Joint_States, Actions


def get_joint_action_list(joint_state, A):
    AA = []
    for idx in range(len(joint_state) - 1):
        s = joint_state[idx]
        AA.append(tuple(A[s]))  # [(Agent1's actions for s1), (Agent2's actions for s2),...]
    # joint_action_A is now a list of tuples of all respective action list for each agent at their respective state
    # now we transform it into joint action tuples which takes 1 action from each agent's tuple and does combinations
    Joint_actions_for_js = list(itertools.product(*AA))  # = [(a1,a2,a3,...an),(a1*,a2*,a3*...),(a1**,a2**,...)]
    return Joint_actions_for_js


############################################################################
# ############## DELETE EVERYTHING BELOW THIS AFTER TESTING ################
############################################################################
#
# def follow_policy(GRID):
#     Pi = copy.copy(GRID.Pi)
#     while not reached_goal(GRID):
#         R = GRID.R(GRID.js, Pi[GRID.js])
#         GRID.Reward += R
#         GRID.trajectory.append(((GRID.js[0], GRID.js[1]), Pi[GRID.js], R))
#         GRID.plan += " -> " + str(Pi[GRID.js])
#         GRID.js = system_do_action(GRID, GRID.js, Pi[GRID.js])
#         GRID.path = GRID.path + "->" + str(GRID.js)
#
#
# Grid = CentralizedEnvironment(1, (1, 1), 'centeralized_baseline/train_grid.txt', 'stochastic', 0.8)
# Grid.V, Grid.Pi = value_iteration(Grid, Grid.S, Grid.A, Grid.R, Grid.gamma)
# follow_policy(Grid)
# for i in Grid.trajectory:
#     print(i)
# print("Reached GOAL: ", Grid.js)

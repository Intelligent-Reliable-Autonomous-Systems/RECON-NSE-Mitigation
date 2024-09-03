import copy
import random
import numpy as np
import read_grid
from typing import List
from sklearn.linear_model import LinearRegression

class SalpEnvironment:
    def __init__(self,  num_of_agents, goal_deposit, grid_filename):
        
        All_States, rows, columns = read_grid.grid_read_from_file(grid_filename)
        self.all_states = copy.copy(All_States)
        self.file_name = grid_filename
        self.sample = ('A', 'B')
        self.weighting = {'X': 0.0, 'A': 2.0, 'B': 5.0}
        self.R_Nmax = max(self.weighting.values())* np.log(num_of_agents)
        self.deposit_goal = goal_deposit  # (the tuple (#A,#B))
        s_goals = np.argwhere(All_States == 'G')
        s_goal = s_goals[0]
        self.s_goal_loc = s_goal[0:2]

        self.num_of_agents = num_of_agents
        self.rows = rows
        self.columns = columns
        self.goal_states = s_goals  # this is a list of list [[gx_1,gy_1],[gx_2,gy_2]...]
        self.All_States = All_States
        self.S = self.get_state_space()
        self.task_list = []
        for i in range(len(self.deposit_goal)):
            for j in range(self.deposit_goal[i]):
                self.task_list.append(self.sample[i])

    def get_state_space(self):
        S = []
        for i in range(self.rows):
            for j in range(self.columns):
                for sample in ['A', 'B', 'X']:
                    for done in [True, False]:
                        S.append((i, j, sample, self.All_States[i][j]=='C', done))
        return S
    
    def get_joint_state(self, Agents):
        joint_state = []
        for agent in Agents:
            joint_state.append(agent.s)
        return tuple(joint_state)

    def all_have_reached_goal(self, Agents):
        for agent in Agents:
            if agent.at_goal() is False:
                return False
        return True
    
    
############################################################################################
############################################################################################
############################################################################################
############################################################################################

class SalpAgent:
    def __init__(self, IDX, Grid:SalpEnvironment, start_location=(0,0)):
        
        # Initialize the agent with environment, sample, start location, and label (to identify incase of multi-agent scenario)
        self.Grid = Grid
        self.start_location = start_location
        self.IDX = IDX
        self.label = str(IDX+1)
        self.goal_loc = Grid.s_goal_loc
        self.assigned_sample = Grid.task_list[IDX]
        self.rows = Grid.rows
        self.columns = Grid.columns
        self.best_performance_flag = False
        
        # Set the success probability of the agent
        self.p_success = 0.8
        
        # set the start state and the goal state in the right format:
        # s = (x, y, sample, coral, done)
        self.s0 = (start_location[0], start_location[1], 'X', Grid.All_States[start_location[0]][start_location[1]]=='C', False)
        self.s = copy.deepcopy(self.s0)
        self.s_goal = (self.goal_loc[0], self.goal_loc[1], 'X', False, True)
        
        # Initialize state and action space
        self.S = Grid.S
        self.A = self.get_action_space()
        self.A_initial = copy.deepcopy(self.A)
        self.R = 0
        self.gamma = 0.99
        
        # Initialize Value function and Policies
        self.V = {}
        self.Pi = {}
        
        # Initialize different policies for the agent
        self.Pi_naive = {}
        self.Pi_recon = {}
        self.Pi_gen_recon_wo_cf = {}
        self.Pi_gen_recon_with_cf = {}
        self.Pi_dr = {}
        self.Pi_considerate = {}
        
        # Initialize the NSE and the NSE tracker
        self.blame_training_data_x_wo_cf = []
        self.blame_training_data_y_wo_cf = []
        self.blame_training_data_x_with_cf = []
        self.blame_training_data_y_with_cf = []
        
        # Initialize the NSE penalty reward functions
        self.R_blame = {}
        self.R_blame_dr = {}
        self.R_blame_gen_with_cf = {}
        self.R_blame_gen_wo_cf = {}
        self.R_blame_considerate = {}
        for s in self.S:
            self.R_blame[s] = 0
            self.R_blame_dr[s] = 0
            self.R_blame_gen_with_cf[s] = 0
            self.R_blame_gen_wo_cf[s] = 0
            self.R_blame_considerate[s] = 0
        
        self.model_wo_cf = LinearRegression()
        self.model_with_cf = LinearRegression()
        
        # variables to track the agent path and trahectory for debuging purposes
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.trajectory = []
    
    
    def Reward(self, s, a):
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,sample_with_agent,coral_flag,done_flag>
        s_next = self.step(s, a)
        if s_next == self.s_goal:
            # print("Agent " + self.label + " reached the goal using " + str(s) + ", " + a + " -> " + str(s_next))
            R = 100
        elif s_next[4] == True:
            R = 0
        else:
            R = -1
        return R
    
    def R_considerate(self, s, a):
        alpha_1 = 0.5
        alpha_2 = 0.5
        s_next = self.step(s, a)
        R = alpha_1 * self.Reward(s, a) + alpha_2 * self.R_blame_considerate[s_next]
        return R
        
    
    def step(self, s, a):
        # state of an agent: <x,y,sample_with_agent,coral_flag,done_flag>
        # operation actions = ['Noop','pick', 'drop', 'U', 'D', 'L', 'R']
        s = list(s)
        if a == 'pick':
            if self.Grid.All_States[s[0]][s[1]] == self.assigned_sample:
                s[2] = self.assigned_sample
            else:
                s[2] = s[2]
        elif a == 'drop':
            if s[0] == self.goal_loc[0] and s[1] == self.goal_loc[1]:
                s[2] = 'X'
                s[4] = True
            else:
                s[2] = s[2]
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            # s = self.move_correctly(self.Grid, s, a)  # can be replaced with a sampling function to incorporate stochasticity
            T = self.get_transition_prob(tuple(s), a)
            # s is the states with the maximum probability
            s = max(T, key=T.get)
        elif a == 'Noop':
            s = s
        else:
            print("INVALID ACTION: ", a)
        s = tuple(s)
        return s

    
    def move_correctly(self, s, a):
        # action = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                s_next = (s[0] - 1, s[1], s[2], self.Grid.All_States[s[0] - 1][s[1]] == 'C', s[4])
        elif a == 'D':
            if s[0] == self.rows - 1:
                s_next = s
            else:
                s_next = (s[0] + 1, s[1], s[2], self.Grid.All_States[s[0] + 1][s[1]] == 'C', s[4])
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                s_next = (s[0], s[1] - 1, s[2], self.Grid.All_States[s[0]][s[1] - 1] == 'C', s[4])
        elif a == 'R':
            if s[1] == self.columns - 1:
                s_next = s
            else:
                s_next = (s[0], s[1] + 1, s[2], self.Grid.All_States[s[0]][s[1] + 1] == 'C', s[4])
        return s_next
    
    def at_goal(self):
        return self.s == self.s_goal
    
    def sample_state(self, s, a):
        '''Sample the next state based on the policy pi and the current state s based on transition probabilities function'''
        p = random.uniform(0, 1)
        # print("Sampling state: ", s)
        # print("Action for sample: ", a)
        T = self.get_transition_prob(s, a)
        cumulative_prob = 0
        for s_prime in list(T.keys()):
            cumulative_prob += T[s_prime]
            if p <= cumulative_prob:
                return s_prime
        return s_prime
    
    def reset(self):
        self.s = copy.deepcopy(self.s0)
        self.path = str(self.s)
        self.trajectory = []
        self.plan = ""
        self.A = copy.deepcopy(self.A_initial)
        return self
        
    def follow_policy(self, Pi=None):
        self.R = 0
        if Pi is None:
            print("No policy provided for agent " + self.label)
            Pi = copy.deepcopy(self.Pi)
        while not self.at_goal():
            # print(str(self.s)+ " -> " + str(Pi[self.s]) + " -> " + str( self.step(self.s, Pi[self.s])))
            R = self.Reward(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            self.s = self.step(self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 40:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop! at state: " + str(self.s))
                    break
        self.trajectory.append((self.s, Pi[self.s], None))
        
    def follow_policy_rollout(self, Pi=None):
        if Pi is None:
            Pi = copy.deepcopy(self.Pi)
        while not self.at_goal():
            # print(str(self.s)+ " -> " + str(Pi[self.s]) + " -> " + str( self.step(self.s, Pi[self.s])))
            R = self.Grid.R1(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            self.s = self.sample_state(self.s, Pi[self.s])  # self.step(self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)
            # if s is stuck in a loop or not making progress, break
            if len(self.trajectory) > 20:
                if self.trajectory[-1] == self.trajectory[-5]:
                    print("Agent " + str(self.label) + " is stuck in a loop!")
                    break
        self.trajectory.append((self.s, Pi[self.s], None))
                    
    def get_action_space(self):
        # Get the action space for salp agent
        A = {}
        Grid = self.Grid
        for s in self.S:
            A[s] = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        # Remove actions that are not possible in certain states
        for s in self.S:
            if s[2] != 'X':
                if 'pick' in A[s]:
                    A[s].remove('pick')
            if s[2] == 'X':
                if 'drop' in A[s]:
                    A[s].remove('drop')
            if self.Grid.All_States[s[0]][s[1]] != self.assigned_sample:
                if 'pick' in A[s]:
                    A[s].remove('pick')
            if s[4] is True:
                if 'pick' in A[s]:
                    A[s].remove('pick')
        return A

    def get_transition_prob(self, s, a):
        # state of an agent: <x, y, onboard_sample, coral_flag, done_flag>
        # operation actions = ['Noop', 'pick', 'drop', 'U', 'D', 'L', 'R']
        p_success = copy.copy(self.p_success)
        p_fail = 1 - p_success
        action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        action_key = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        if s == self.s_goal or a == 'Noop':
            T = {s: 1}  # stay at the goal with prob = 1
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s_next_correct = self.move_correctly(s, a)
            s_next_slide_left = self.move_correctly(s, action[(action_key[a] - 1) % 4])
            s_next_slide_right = self.move_correctly(s, action[(action_key[a] + 1) % 4])
            if s_next_correct == s_next_slide_left:
                T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_right: round(p_fail / 2, 3)}
            elif s_next_correct == s_next_slide_right:
                T = {s_next_correct: round(p_success + p_fail / 2, 3), s_next_slide_left: round(p_fail / 2, 3)}
            else:
                T = {s_next_correct: round(p_success, 3),
                    s_next_slide_left: round(p_fail / 2, 3),
                    s_next_slide_right: round(p_fail / 2, 3)}
        else:
            T = {self.step(s, a): 1} 
        return T

    def generalize_Rblame_linearReg(self):
        weighting = self.Grid.weighting
        X_wo_cf = copy.deepcopy(self.blame_training_data_x_wo_cf)
        y_wo_cf = copy.deepcopy(self.blame_training_data_y_wo_cf)
        X_with_cf = copy.deepcopy(self.blame_training_data_x_with_cf)
        y_with_cf = copy.deepcopy(self.blame_training_data_y_with_cf)
        N1 = len(X_wo_cf)
        N2 = len(X_with_cf)
        X1, y1 = np.array(X_wo_cf).reshape((N1, 4)), np.array(y_wo_cf).reshape((N1, 1))
        X2, y2 = np.array(X_with_cf).reshape((N2, 4)), np.array(y_with_cf).reshape((N2, 1))

        #####################################
        # # Saving training data as text files
        # filename_X1 = 'training_data/Agent' + self.label + '_X_wo_cf.txt'
        # filename_y1 = 'training_data/Agent' + self.label + '_y_wo_cf.txt'
        # filename_X2 = 'training_data/Agent' + self.label + '_X_with_cf.txt'
        # filename_y2 = 'training_data/Agent' + self.label + '_y_with_cf.txt'
        #
        # np.savetxt(filename_X1, X1)
        # np.savetxt(filename_y1, y1)
        # np.savetxt(filename_X2, X2)
        # np.savetxt(filename_y2, y2)
        #####################################

        model_wo_cf_data = LinearRegression()
        model_with_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        model_with_cf_data.fit(X2, y2)
        # s = (x, y, sample, coral, done)
        self.model_wo_cf = model_wo_cf_data
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

        # print("%%%%%%%%%%[init_agent.py] RBlame1 == RBlame1: ", self.R_blame_gen_wo_cf == self.R_blame_gen_wo_cf)
        # print("%%%%%%%%%%[init_agent.py] RBlame1 == RBlame2: ", self.R_blame_gen_wo_cf == self.R_blame_gen_with_cf)
        self.R_blame_gen_wo_cf[self.s_goal] = 100
        self.R_blame_gen_with_cf[self.s_goal] = 100

    def generalize_Rblame_wo_cf(self):
        weighting = self.Grid.weighting
        X_wo_cf = copy.deepcopy(self.blame_training_data_x_wo_cf)
        y_wo_cf = copy.deepcopy(self.blame_training_data_y_wo_cf)
        N1 = len(X_wo_cf)
        X1, y1 = np.array(X_wo_cf).reshape((N1, 2)), np.array(y_wo_cf).reshape((N1, 1))

        model_wo_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        self.model_wo_cf = model_wo_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)

        self.R_blame_gen_wo_cf[self.s_goal] = 100

    def generalize_Rblame_with_cf(self):
        weighting = self.Grid.weighting
        X_with_cf = copy.deepcopy(self.blame_training_data_x_with_cf)
        y_with_cf = copy.deepcopy(self.blame_training_data_y_with_cf)
        N2 = len(X_with_cf)
        X2, y2 = np.array(X_with_cf).reshape((N2, 2)), np.array(y_with_cf).reshape((N2, 1))

        model_with_cf_data = LinearRegression()
        model_with_cf_data.fit(X2, y2)
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

        self.R_blame_gen_with_cf[self.s_goal] = 100

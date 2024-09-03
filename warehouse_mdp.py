import copy
import random
import numpy as np
import read_grid
from typing import List
from sklearn.linear_model import LinearRegression

class WarehouseEnvironment:
    def __init__(self,  num_of_agents, goal_deposit, grid_filename):
        
        All_States, rows, columns = read_grid.grid_read_from_file(grid_filename)
        self.all_states = copy.copy(All_States)
        self.file_name = grid_filename
        self.shelf = ('1', '2')
        self.weighting = {'X': 0, 's1': 2, 's2': 2, 'S1': 5, 'S2': 5}
        self.R_Nmax = max(self.weighting.values())* np.log(num_of_agents + 1)
        self.deposit_goal = goal_deposit  # (the tuple (#s1,#s2))
        self.goal_locs = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'g']
        self.shelf_assigner = {'1':[], '2':[]}
        self.shelf_assigner['1'] = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 't' and j < columns//2]
        self.shelf_assigner['2'] = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 't' and j >= columns//2]
        
        self.num_of_agents = num_of_agents
        self.rows = rows
        self.columns = columns
        self.All_States = All_States
        self.S = self.get_state_space()
        self.task_list = []
        for i in range(len(self.deposit_goal)):
            for j in range(self.deposit_goal[i]):
                self.task_list.append(self.shelf[i])

    def get_state_space(self):
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        S = []
        for i in range(self.rows):
            for j in range(self.columns):
                for shelf in self.weighting.keys():
                    for done in [True, False]:
                        S.append((i, j, shelf, self.All_States[i][j]=='s', done))
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

class WarehouseAgent:
    def __init__(self, IDX, Grid:WarehouseEnvironment, start_location=(0,0)):
        
        # Initialize the agent with environment, sample, start location, and label (to identify incase of multi-agent scenario)
        self.Grid = Grid
        self.start_location = start_location
        self.IDX = IDX
        self.label = str(IDX+1)
        self.goal_loc = Grid.goal_locs[IDX%len(Grid.goal_locs)]
        self.assigned_shelf = 's'+Grid.task_list[IDX]
        self.shelf_loc = Grid.shelf_assigner[Grid.task_list[IDX]][IDX%len(Grid.shelf_assigner[Grid.task_list[IDX]])]
        self.rows = Grid.rows
        self.columns = Grid.columns
        self.best_performance_flag = False
        
        # Set the success probability of the agent
        self.p_success = 0.8
        
        # set the start state and the goal state in the right format:
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        self.s0 = (start_location[0], start_location[1], 'X', Grid.All_States[start_location[0]][start_location[1]] == 's', False)
        self.s = copy.deepcopy(self.s0)
        self.s_goal = (self.shelf_loc[0], self.shelf_loc[1], 'X', False, True)
        
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
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        # actions = ['Noop', 'toggle_load', 'U', 'D', 'L', 'R']
        s_next = self.step(s, a)
        if s_next == self.s_goal:
            R = 100
        elif s[4] == True:
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
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        # operation actions = ['Noop', 'toggle_load', 'U', 'D', 'L', 'R']
        s_next = list(s)
        if a == 'toggle_load':
            if s[:2] == self.shelf_loc and s[2] == 'X' and s[4] is False:
                s_next[2] = self.assigned_shelf  # loading the shelf from its location
            elif s[:2] == self.goal_loc and self.Grid.All_States[s[0]][s[1]] == 'g' and s[2] == self.assigned_shelf:
                s_next[2] = self.get_processed_shelf_type()
            elif (s[:2] == self.shelf_loc and s[2] == self.get_processed_shelf_type()):
                s_next[2] = 'X'  # drop the process shelf back to its location
                s_next[4] = True
        elif a == 'U' or a == 'D' or a == 'L' or a == 'R':
            s_next = self.move_correctly(s, a)  
        elif a == 'Noop':
            s_next = copy.deepcopy(s)
        else:
            print("INVALID ACTION: ", a)
        s_next = tuple(s_next)
        return s_next
    
    def move_correctly(self, s, a):
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        s_next = 0
        if a == 'U':
            if s[0] == 0:
                s_next = s
            else:
                s_next = (s[0] - 1, s[1], s[2], self.Grid.All_States[s[0] - 1][s[1]] == 's', s[4])
        elif a == 'D':
            if s[0] == self.rows - 1:
                s_next = s
            else:
                s_next = (s[0] + 1, s[1], s[2], self.Grid.All_States[s[0] + 1][s[1]] == 's', s[4])
        elif a == 'L':
            if s[1] == 0:
                s_next = s
            else:
                s_next = (s[0], s[1] - 1, s[2], self.Grid.All_States[s[0]][s[1] - 1] == 's', s[4])
        elif a == 'R':
            if s[1] == self.columns - 1:
                s_next = s
            else:
                s_next = (s[0], s[1] + 1, s[2], self.Grid.All_States[s[0]][s[1] + 1] == 's', s[4])
        else:
            print("NOT A VALID MOVE ACTION: ", a)
            s_next = s
        return s_next
    
    def at_goal(self):
        return self.s == self.s_goal
    
    def get_processed_shelf_type(self):
        # s1 turns to S1 and s2 turns to S2
        if self.assigned_shelf in ['s1', 's2']:
            return self.assigned_shelf.upper()
        else:
            print("ERROR: Shelf type is not s1 or s2: ", self.assigned_shelf)
            
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
        # Get the action space for Warehouse agent
        A = {}
        Grid = self.Grid
        for s in self.S:
            A[s] = ['Noop', 'toggle_load', 'U', 'D', 'L', 'R']
        return A

    def get_transition_prob(self, s, a):
        s_next = self.step(s, a)
        if s == self.s_goal:
            T = {s: 1}
        else:
            T = {s:0.2, s_next: 0.8}
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
        model_wo_cf_data = LinearRegression()
        model_with_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        model_with_cf_data.fit(X2, y2)
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        self.model_wo_cf = model_wo_cf_data
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3])]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)
        self.R_blame_gen_wo_cf[self.s_goal] = 100
        self.R_blame_gen_with_cf[self.s_goal] = 100

    def generalize_Rblame_wo_cf(self):
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
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
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
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

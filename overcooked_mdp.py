import copy
import random
import numpy as np
import read_grid
from typing import List
from sklearn.linear_model import LinearRegression

class OvercookedEnvironment:
    def __init__(self,  num_of_agents, goal_deposit, grid_filename):
        
        All_States, rows, columns = read_grid.grid_read_from_file(grid_filename)
        self.all_states = copy.copy(All_States)
        self.file_name = grid_filename
        self.dish = ('O', 'T')
        self.weighting = {
            "X": 0.0,  # nothing in hand
            "Dt": 2.0,  # ready tomato dish in hand
            "Do": 5.0,  # ready onion dish in hand
            "T": 2.0,  # tomato in hand
            "O": 5.0,  # onion in hand
            "t": 0.0,  # tomato in pot
            "o": 0.0,  # onion in pot
            "D": 5.0,  # dish in hand
        }
        self.looking_dir = {  # 0: up, 1: right, 2: down, 3: left in row, column format
            0: [-1, 0],
            1: [0, +1],
            2: [+1, 0],
            3: [0, -1],
        } 
        
        self.serving_counter_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'S']
        self.pot_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'P']
        self.dustbin_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'W']  # W for wastebin since D is used for dish
        self.onion_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'O']
        self.tomato_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'T']
        self.dish_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'D']
        self.normal_counter_locations = [(i,j) for i in range(rows) for j in range(columns) if All_States[i][j] == 'X']
        # goal locations are the ones where the agent can look down towards the serving counter therefore, the goal locations are one row above the serving counter locations
        self.s_goal_locs = [(serving_counter_location[0]-1, serving_counter_location[1]) for serving_counter_location in self.serving_counter_locations]
        
        self.R_Nmax = max(self.weighting.values())* np.log(num_of_agents+1)
        self.deposit_goal = goal_deposit  # (the tuple (#O,#T) sum of onion soups, tomato soups, and cleaner agents that are assigned later
        self.num_of_agents = num_of_agents
        self.rows = rows
        self.columns = columns
        self.All_States = All_States
        self.S = self.get_state_space()
        self.task_list = []
        for i in range(len(self.deposit_goal)):
            for j in range(self.deposit_goal[i]):
                self.task_list.append(self.dish[i])
        # remove 20% of the tasks with an equal mix of O and T replaced by C that represent the cleaner agents
        num_of_cleaners = int(0.2 * len(self.task_list))
        task_list = copy(self.task_list)
        for i in range(num_of_cleaners):
            task_list[i] = 'C'
        random.shuffle(task_list)

    def get_state_space(self):
        # s = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        S = []
        for i in range(1,self.rows-1):
            for j in range(1,self.columns-1):
                for dir in range(4):
                    for object_in_hand in list(self.weighting.keys()):
                        for done in [True, False]:
                            S.append((i, j, dir, self.All_States[i][j]=='W', object_in_hand, done))
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

class OvercookedAgent:
    def __init__(self, IDX, Grid:OvercookedEnvironment, start_location=(1,1)):
        
        # Initialize the agent with environment, sample, start location, and label (to identify incase of multi-agent scenario)
        self.Grid = Grid
        self.start_location = start_location
        self.IDX = IDX
        self.label = str(IDX+1)
        self.goal_loc = Grid.s_goal_locs[IDX%len(self.Grid.s_goal_locs)]
        self.assigned_dish = Grid.task_list[IDX]
        self.assigned_pot_loc = Grid.pot_locations[IDX%len(Grid.pot_locations)]
        self.rows = Grid.rows
        self.columns = Grid.columns
        self.best_performance_flag = False
        
        if self.assigned_dish == 'C':  # cleaner agent
            self.goal_loc = Grid.dustbin_locations[self.IDX%len(Grid.dustbin_locations)]
            
        # Set the success probability of the agent
        self.p_success = 0.8
        
        # set the start state and the goal state in the right format:
        # s = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        # self.looking_dir = {  # 0: up, 1: right, 2: down, 3: left in row, column format
        #     0: [-1, 0], # looking up to the abilve row
        #     1: [0, +1], # looking right to the next column
        #     2: [+1, 0], # looking down to the next row
        #     3: [0, -1],}  # looking left to the previous column
        self.s0 = (start_location[0], start_location[1], 2, self.Grid.All_States[start_location[0]][start_location[1]] == 'W', 'X', False)
        self.s = copy.deepcopy(self.s0)
        self.s_goal = (self.goal_loc[0], self.goal_loc[1], 2, self.Grid.All_States[self.goal_loc[0]][self.goal_loc[1]] == 'W', 'X', True)
            
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
        # operation actions = ['Noop', 'interact', 'forward', 'turn_left', 'turn_right']
        # s = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        s_next = self.step(s, a)
        if s_next == self.s_goal:
            R = 100
        else:
            R = -1
        return R
    
    def R_considerate(self, s, a):
        alpha_1 = 0.5
        alpha_2 = 0.5
        s_next = self.step(s, a)
        R = alpha_1 * self.Reward(s, a) + alpha_2 * self.R_blame_considerate[s_next]
        return R
        
    def get_action_space(self):
        # Get the action space for salp agent
        A = {}
        Grid = self.Grid
        for s in self.S:
            A[s] = ['Noop', 'interact', 'forward', 'turn_left', 'turn_right']
        # Remove actions that are not possible in certain states
        #    --> for now, no need to forbit any action.
        return A

    def is_looking_at(self, s, entity):
        # s = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        # self.looking_dir = {  # 0: up, 1: right, 2: down, 3: left in row, column format
        #     0: [-1, 0],
        #     1: [0, +1],
        #     2: [+1, 0],
        #     3: [0, -1],} 
        if self.Grid.All_States[s[0]+self.Grid.looking_dir[s[2]][0]][s[1]+self.Grid.looking_dir[s[2]][1]] == entity:
            return True
        return False
    
    def is_looking_at_loc(self, s, loc):
        if (s[0]+self.Grid.looking_dir[s[2]][0], s[1]+self.Grid.looking_dir[s[2]][1]) == loc:
            return True
        return False
    
    def step(self, state, action):
        s = copy.deepcopy(state)
        s_next = list(copy.deepcopy(state))
        # state = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        #    -> s[3]: direction = [0, 1, 2, 3]  # 0: up, 1: right, 2: down, 3: left
        # action = {'Noop', 'interact', 'forward', 'turn_left', 'turn_right'}
        motion = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        look_left = {0: 3, 1: 0, 2: 1, 3: 2}
        look_right = {0: 1, 1: 2, 2: 3, 3: 0}

        if action == 'Noop':
            s_next = list(copy.deepcopy(state))
        elif action == 'forward':
            # forward motion into the direction of the agent is looking at, however, the agent
            # stays there if it is looking at any of the edges of the grid of a pot 'P'
            # state = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
            if s[2] == 0 and s[0] == 1:
                s_next = list(copy.deepcopy(state))
            elif s[2] == 1 and s[1] == self.columns - 2:
                s_next = list(copy.deepcopy(state))
            elif s[2] == 2 and s[0] == self.rows - 2:
                s_next = list(copy.deepcopy(state))
            elif s[2] == 3 and s[1] == 1:
                s_next = list(copy.deepcopy(state))
            elif self.is_looking_at(s, 'P'):
                s_next = list(copy.deepcopy(state))
            else:
                # s = < s[0]: j,s[1]: i,s[2]: dir,s[3]: water_flag ,s[4]: in_hand_object,s[5]: done>
                s_next[0] += motion[s[2]][0]
                s_next[1] += motion[s[2]][1]
                s_next[3] = self.Grid.All_States[s_next[0]][s_next[1]] == "W"
        elif action == 'turn_left':
            s_next[2] = look_left[s_next[2]]
        elif action == 'turn_right':
            s_next[2] = look_right[s_next[2]]
        elif action == 'interact':
            # s = < s[0]: j,s[1]: i,s[2]: dir,s[3]: water_flag ,s[4]: in_hand_object,s[5]: done>
            # s[4] = {'X', 'Dt', 'Do', 'T', 'O', 't', 'o', 'D'}
            # dish cooking procedure: 
            # 'X' -(collect raw O/T)-> 'O/T' -(put it in pot)-> 'o/t' -(go to collect a dish)-> 'D'...
            # ... D -(go to pot and put cooked stuff in dish)-> 'Do/t' -(deliver at serving counter)-> 'X'
            if self.is_looking_at(s, self.assigned_dish) and s[4] == 'X':
                s_next[4] = self.assigned_dish
                # print("[interact 1] s_next: ", tuple(s_next))
            elif self.is_looking_at_loc(s, self.assigned_pot_loc) and self.is_looking_at(s,'P') and s[4] == self.assigned_dish:
                s_next[4] = self.assigned_dish.lower()
                # print("[interact 2] s_next: ", tuple(s_next))
            elif self.is_looking_at(s, 'D') and s[4] == self.assigned_dish.lower():
                s_next[4] = 'D'
                # print("[interact 3] s_next: ", tuple(s_next))
            elif self.is_looking_at_loc(s, self.assigned_pot_loc) and self.is_looking_at(s,'P') and s[4] == 'D':
                s_next[4] = 'D' + self.assigned_dish.lower()
                # print("[interact 4] s_next: ", tuple(s_next))
            elif self.is_looking_at(s, 'S') and s[4] == 'D' + self.assigned_dish.lower() and s[5] == False:
                s_next[4] = "X"
                s_next[5] = True
                # print("[interact 5] s_next: ", tuple(s_next))
        s_next = tuple(s_next)

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
                    

    def get_transition_prob(self, s, a):
        # state = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        # operation actions = ['Noop', 'interact', 'forward', 'turn_left', 'turn_right']
        p_success = copy.copy(self.p_success)
        p_fail = 1 - p_success
        s_next = self.step(s, a)
        if s_next == self.s_goal:
            T = {s_next: 1}
        else:
            T = {s: 0.2, s_next: 0.8} 
        return T

    def generalize_Rblame_linearReg(self):
        weighting = self.Grid.weighting
        X_wo_cf = copy.deepcopy(self.blame_training_data_x_wo_cf)
        y_wo_cf = copy.deepcopy(self.blame_training_data_y_wo_cf)
        X_with_cf = copy.deepcopy(self.blame_training_data_x_with_cf)
        y_with_cf = copy.deepcopy(self.blame_training_data_y_with_cf)
        N1 = len(X_wo_cf)
        N2 = len(X_with_cf)
        X1, y1 = np.array(X_wo_cf).reshape((N1, 2)), np.array(y_wo_cf).reshape((N1, 1))
        X2, y2 = np.array(X_with_cf).reshape((N2, 2)), np.array(y_with_cf).reshape((N2, 1))
        model_wo_cf_data = LinearRegression()
        model_with_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        model_with_cf_data.fit(X2, y2)
        # s = (s[0]: row, s[1]: column, s[2]: direction, s[3]: dustbin, s[4]: object in hand, s[5]: done)
        self.model_wo_cf = model_wo_cf_data
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[int(s[3]), weighting[s[4]]]])))
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[int(s[3]), weighting[s[4]]]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

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
                self.model_wo_cf.predict(np.array([[int(s[3]), weighting[s[4]]]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)

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
                self.model_with_cf.predict(np.array([[int(s[3]), weighting[s[4]]]])))
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

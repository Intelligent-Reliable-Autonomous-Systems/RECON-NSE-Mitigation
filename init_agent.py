import copy

import calculation_lib
import init_env
from calculation_lib import do_action
import numpy as np
from sklearn.linear_model import LinearRegression


class Agent:
    def __init__(self, start_loc, Grid, label):
        self.goal_modes = Grid.goal_modes
        self.p_success = Grid.p_success
        self.label = label
        self.IDX = int(label) - 1
        # state of an agent: <x,y,box_with_me,coral_flag,box_at_goal>
        coral_flag = Grid.coral_flag[start_loc[0]][start_loc[1]]
        self.s = (start_loc[0], start_loc[1], 'X', coral_flag, self.goal_modes[self.IDX])
        self.s0 = (start_loc[0], start_loc[1], 'X', coral_flag, self.goal_modes[self.IDX])
        self.startLoc = start_loc
        self.best_performance_flag = False
        self.goal_states = Grid.goal_states
        self.s_goal = Grid.s_goal
        self.Grid = Grid
        self.P = 1.0
        self.R = 0.0
        self.gamma = 0.95
        self.V = []
        self.Pi = []
        self.NSE = 0.0
        self.NSE_gen = 0.0
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.trajectory = []
        self.num_of_agents = Grid.num_of_agents
        self.done_flag = False
        # print('[from init_agent] Goal Mode for Agent ' + self.label + ' at initialization is: ' + str(self.s0[4]))

        # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        self.A = {}
        # counterfactual comparison actions = ['switch_compare']
        self.A2 = {}
        self.R_blame = {}
        self.blame_training_data_x = []
        self.blame_training_data_y = []
        self.R_blame_gen = {}
        self.model = LinearRegression()
        for s in Grid.S:
            self.A[s] = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']  # operation actions
            self.A2[s] = ['switch_compare']  # action for counterfactuals comparison
            self.R_blame[s] = 0.0
            self.R_blame_gen[s] = 0.0
        for s in self.Grid.S:
            if s[2] != 'X':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if s[2] == 'X':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if self.Grid.All_States[s[0]][s[1]] != 'J' and Grid.All_States[s[0]][s[1]] != 'j':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if self.Grid.All_States[s[0]][s[1]] != 'G':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if s[4][0] >= self.s_goal[4][0]:
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
            if s[4][1] >= self.s_goal[4][1]:
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if s[2] == 'S' and s[4][0] > self.s_goal[4][0]:
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if s[2] == 'L' and s[4][1] > self.s_goal[4][1]:
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if self.Grid.All_States[s[0]][s[1]] == 'j':
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if self.Grid.All_States[s[0]][s[1]] == 'J':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')

    def Reward(self, s, a):
        # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,trash_box_size,coral_flag,list_of_boxes_at_goal>
        if calculation_lib.do_action(self, s, a) == self.s_goal:
            if a == 'drop':
                R = 100
            else:
                R = -1
        else:
            R = -1
        return R

    def follow_policy(self):
        Pi = copy.copy(self.Pi)


        while self.s != self.s_goal:
            R = self.Reward(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            # print("Action: ", Pi[self.s])
            # print("Plan till here: ", self.plan)
            self.s = do_action(self, self.s, Pi[self.s])
            # x, y, _, _, _ = self.s
            # loc = (x, y)
            self.path = self.path + "->" + str(self.s)

    def agent_reset(self):
        self.NSE = 0.0
        self.s = copy.deepcopy(self.s0)
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.R = 0.0

    def generalize_Rblame_linearReg(self):
        weighting = {'X': 0.0, 'S': 3.0, 'L': 10.0}
        X = copy.deepcopy(self.blame_training_data_x)
        N = len(X)
        y = copy.deepcopy(self.blame_training_data_y)
        X, y = np.array(X).reshape((N, 4)), np.array(y).reshape((N, 1))
        model = LinearRegression()
        model.fit(X, y)
        self.model = model
        for s in self.Grid.S:
            R_blame_prediction_for_s = float(self.model.predict(np.array([[weighting[s[2]], int(s[3]), s[4][0], s[4][1]]])))
            self.R_blame_gen[s] = round(R_blame_prediction_for_s, 1)

        self.R_blame_gen[self.s_goal] = 100



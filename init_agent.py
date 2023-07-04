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
        self.blame_training_data_x_wo_cf = []
        self.blame_training_data_y_wo_cf = []
        self.blame_training_data_x_with_cf = []
        self.blame_training_data_y_with_cf = []
        self.R_blame_gen_with_cf = {}
        self.R_blame_gen_wo_cf = {}
        self.model_wo_cf = LinearRegression()
        self.model_with_cf = LinearRegression()
        for s in Grid.S:
            self.A[s] = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']  # operation actions
            self.A2[s] = ['switch_compare']  # action for counterfactuals comparison
            self.R_blame[s] = 0.0
            self.R_blame_gen_with_cf[s] = 0.0
            self.R_blame_gen_wo_cf[s] = 0.0
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
        X_wo_cf = copy.deepcopy(self.blame_training_data_x_wo_cf)
        y_wo_cf = copy.deepcopy(self.blame_training_data_y_wo_cf)
        X_with_cf = copy.deepcopy(self.blame_training_data_x_with_cf)
        y_with_cf = copy.deepcopy(self.blame_training_data_y_with_cf)
        N1 = len(X_wo_cf)
        N2 = len(X_with_cf)
        print("-----------------[init_agent.py] len(X_wo_cf)", N1)
        print("-----------------[init_agent.py] len(X_with_cf)", N2)
        X1, y1 = np.array(X_wo_cf).reshape((N1, 4)), np.array(y_wo_cf).reshape((N1, 1))
        X2, y2 = np.array(X_with_cf).reshape((N2, 4)), np.array(y_with_cf).reshape((N2, 1))
        print("-----------------[init_agent.py] len(X1)", len(X1))
        print("-----------------[init_agent.py] len(y1)", len(y1))
        print("-----------------[init_agent.py] len(X2)", len(X2))
        print("-----------------[init_agent.py] len(y2)", len(y2))

        model_wo_cf_data = LinearRegression()
        model_with_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        model_with_cf_data.fit(X2, y2)
        print("=================[init_agent.py] model1 == model1: ", model_wo_cf_data == model_wo_cf_data)
        print("=================[init_agent.py] model1 == model2: ", model_wo_cf_data == model_with_cf_data)
        self.model_wo_cf = model_wo_cf_data
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3]), s[4][0], s[4][1]]])))
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3]), s[4][0], s[4][1]]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

        print("%%%%%%%%%%[init_agent.py] RBlame1 == RBlame1: ", self.R_blame_gen_wo_cf == self.R_blame_gen_wo_cf)
        print("%%%%%%%%%%[init_agent.py] RBlame1 == RBlame2: ", self.R_blame_gen_wo_cf == self.R_blame_gen_with_cf)
        self.R_blame_gen_wo_cf[self.s_goal] = 100
        self.R_blame_gen_with_cf[self.s_goal] = 100

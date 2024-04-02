import copy

import calculation_lib
import init_env
from calculation_lib import do_action
import numpy as np
from sklearn.linear_model import LinearRegression


class Agent:
    def __init__(self, start_loc, Grid, label, assigned_sample):
        self.startLoc = start_loc
        self.Grid = Grid
        self.label = label
        self.IDX = int(label) - 1
        self.assigned_sample = assigned_sample
        self.done_flag = False
        self.p_success = Grid.p_success
        coral_flag = Grid.coral_flag[start_loc[0]][start_loc[1]]

        self.s = (start_loc[0], start_loc[1], 'X', coral_flag, False)
        self.s0 = (start_loc[0], start_loc[1], 'X', coral_flag, False)
        self.s_goal = (Grid.s_goal_loc[0], Grid.s_goal_loc[1], 'X', False, True)

        self.best_performance_flag = False
        self.goal_states = Grid.goal_states
        self.s_goal_loc = Grid.s_goal_loc
        self.P = 1.0
        self.R = 0.0
        self.gamma = 0.95
        self.V = []
        self.Pi = []
        self.NSE = 0.0
        self.NSE_gen = 0.0
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.plan_from_all_methods = []
        self.trajectory = []
        self.num_of_agents = Grid.num_of_agents

        # operation actions = ['Noop', 'pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
        self.A = {}
        # counterfactual comparison actions = ['switch_compare']
        self.A2 = {}
        self.blame_training_data_x_wo_cf = []
        self.blame_training_data_y_wo_cf = []
        self.blame_training_data_x_with_cf = []
        self.blame_training_data_y_with_cf = []
        self.R_blame = {}
        self.R_blame_dr = {}
        self.R_blame_gen_with_cf = {}
        self.R_blame_gen_wo_cf = {}
        self.R_blame_considerate = {}
        self.model_wo_cf = LinearRegression()
        self.model_with_cf = LinearRegression()

        # state of an agent: <x,y,sample_with_me,coral_flag,done_flag>
        for s in Grid.S:
            self.A[s] = ['Noop', 'pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']  # operation actions
            self.A2[s] = ['switch_compare']  # action for counterfactuals comparison
            self.R_blame[s] = 0.0
            self.R_blame_gen_with_cf[s] = 0.0
            self.R_blame_gen_wo_cf[s] = 0.0
            self.R_blame_dr[s] = 0.0
            self.R_blame_considerate[s] = 0.0
        for s in self.Grid.S:
            if self.assigned_sample == 'A':
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_B')
            if self.assigned_sample == 'B':
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_A')
            if s != (self.s_goal_loc[0], self.s_goal_loc[1], self.assigned_sample, False, False):
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if (s[0], s[1]) != tuple(Grid.s_goal_loc):  # 'Noop' is only allowed when agent task is done
                if 'Noop' in self.A[s]:
                    self.A[s].remove('Noop')
            if s[2] != 'X':
                if 'pick_A' in self.A[s]:
                    self.A[s].remove('pick_A')
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_B')
            if s[2] == 'X':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if self.Grid.All_States[s[0]][s[1]] != 'A' and Grid.All_States[s[0]][s[1]] != 'B':
                if 'pick_A' in self.A[s]:
                    self.A[s].remove('pick_A')
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_B')
            if self.Grid.All_States[s[0]][s[1]] != 'G':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if s[4] is True:
                if 'pick_A' in self.A[s]:
                    self.A[s].remove('pick_A')
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_B')
            if self.Grid.All_States[s[0]][s[1]] == 'A':
                if 'pick_B' in self.A[s]:
                    self.A[s].remove('pick_B')
            if self.Grid.All_States[s[0]][s[1]] == 'B':
                if 'pick_A' in self.A[s]:
                    self.A[s].remove('pick_A')

    def Reward(self, s, a):
        # operation actions = ['pick_A', 'pick_B', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,sample_with_agent,coral_flag,done_flag>
        if s[0:2] == self.s_goal[0:2]:
            if a == 'drop' and s[4] is False:
                R = 100
            elif s[4] is True:
                R = 0
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
            self.s = do_action(self, self.s, Pi[self.s])
            self.path = self.path + "->" + str(self.s)

    def agent_reset(self):
        self.NSE = 0.0
        self.s = copy.deepcopy(self.s0)
        self.path = str(self.s)  # + "->"
        self.plan_from_all_methods.append(self.plan)
        self.plan = ""
        self.R = 0.0

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
        # print("=================[init_agent.py] model1 == model1: ", model_wo_cf_data == model_wo_cf_data)
        # print("=================[init_agent.py] model1 == model2: ", model_wo_cf_data == model_with_cf_data)
        self.model_wo_cf = model_wo_cf_data
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3]), int(s[4])]])))
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3]), int(s[4])]])))
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
        X1, y1 = np.array(X_wo_cf).reshape((N1, 3)), np.array(y_wo_cf).reshape((N1, 1))

        model_wo_cf_data = LinearRegression()
        model_wo_cf_data.fit(X1, y1)
        self.model_wo_cf = model_wo_cf_data
        for s in self.Grid.S:
            R_blame_gen_wo_cf = float(
                self.model_wo_cf.predict(np.array([[weighting[s[2]], int(s[3]), int(s[4])]])))
            self.R_blame_gen_wo_cf[s] = round(R_blame_gen_wo_cf, 1)

        self.R_blame_gen_wo_cf[self.s_goal] = 100

    def generalize_Rblame_with_cf(self):
        weighting = self.Grid.weighting
        X_with_cf = copy.deepcopy(self.blame_training_data_x_with_cf)
        y_with_cf = copy.deepcopy(self.blame_training_data_y_with_cf)
        N2 = len(X_with_cf)
        X2, y2 = np.array(X_with_cf).reshape((N2, 3)), np.array(y_with_cf).reshape((N2, 1))

        model_with_cf_data = LinearRegression()
        model_with_cf_data.fit(X2, y2)
        self.model_with_cf = model_with_cf_data
        for s in self.Grid.S:
            R_blame_gen_with_cf = float(
                self.model_with_cf.predict(np.array([[weighting[s[2]], int(s[3]), int(s[4])]])))
            self.R_blame_gen_with_cf[s] = round(R_blame_gen_with_cf, 1)

        self.R_blame_gen_with_cf[self.s_goal] = 100

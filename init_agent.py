import copy
import init_env
from calculation_lib import move

s_goals = init_env.s_goals


class Agent:
    def __init__(self, start_loc, Grid, label):
        # state of an agent: <x,y,box_with_me,coral_flag,box_at_goal>
        self.s = (start_loc[0], start_loc[1], 'None', Grid.coral_flag[start_loc[0]][start_loc[1]], tuple('None'))
        self.s0 = (start_loc[0], start_loc[1], 'None', Grid.coral_flag[start_loc[0]][start_loc[1]], tuple('None'))
        self.startLoc = start_loc
        self.best_performance_flag = False
        self.goal_states = Grid.goal_states
        self.Grid = Grid
        self.P = 1.0
        self.R = 0.0
        self.gamma = 0.99
        self.V = []
        self.Pi = []
        self.NSE = 0.0
        self.label = label
        self.path = str(self.s)  # + "->"

        # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        self.A = {}
        # counterfactual comparison actions = ['switch_compare']
        self.A2 = {}
        self.R_blame = {}
        for s in Grid.S:
            self.A[s] = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']  # operation actions
            self.A2[s] = ['switch_compare']  # action for counterfactuals comparison
            self.R_blame[s] = 0.0
        for s in Grid.S:
            if s[0] == 0:
                if 'U' in self.A[s]:
                    self.A[s].remove('U')
            if s[0] == Grid.rows - 1:
                if 'D' in self.A[s]:
                    self.A[s].remove('D')
            if s[1] == 0:
                if 'L' in self.A[s]:
                    self.A[s].remove('L')
            if s[1] == Grid.columns - 1:
                if 'R' in self.A[s]:
                    self.A[s].remove('R')
            if s[2] != 'None':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if s[2] == 'None':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if len(s[4]) == 2:
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if Grid.All_States[s[0]][s[1]] != 'J':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if Grid.All_States[s[0]][s[1]] != 'G':
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
        for s in Grid.S:
            if len(s[4]) == 2 and s[2] != 'None':
                del Grid.S[s]
            if len(s[4]) == 2:
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')

    def Reward(self, s, a):
        # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,trash_box_size,coral_flag,list_of_boxes_at_goal>
        if s == (self.Grid.goal_states[0][0], self.Grid.goal_states[0][1], 'L', False, ['S']) or s == (
                self.Grid.goal_states[0][0], self.Grid.goal_states[0][1], 'S', False, ['L']):
            if a == 'drop':
                R = 100
            else:
                R = -1
        else:
            R = -1
        return R


def agent_reset(self):
    self.NSE = 0.0
    self.s = copy.deepcopy(self.s0)
    self.path = str(self.s)  # + "->"
    self.R = 0.0


def follow_policy(self, Grid):
    Pi = copy.copy(self.Pi)
    while Pi[self.s] != 'G':
        self.R += Grid.R[self.s]
        self.NSE += Grid.NSE[self.s]
        self.s = move(self.s, Pi[self.s])
        x, y, _ = self.s
        loc = (x, y)
        self.path = self.path + "->" + str(loc)

    # add the Goal reward
    self.R += Grid.R[self.s]

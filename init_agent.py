import copy
import init_env
from calculation_lib import move

s_goals = init_env.s_goals


class Agent:
    def __init__(self, start_loc, Grid, trash_unit_size, label):
        # state of an agent: <x,y,junk_size,coral_flag>
        self.s = (start_loc[0], start_loc[1], trash_unit_size, Grid.coral_flag[start_loc[0]][start_loc[1]])
        self.s0 = (start_loc[0], start_loc[1], trash_unit_size, Grid.coral_flag[start_loc[0]][start_loc[1]])
        self.startLoc = start_loc
        self.best_performance_flag = False

        # self.A = ['U', 'D', 'L', 'R']
        self.A = {}
        self.R_blame = {}
        for s in Grid.S:
            self.A[s] = ['U', 'D', 'L', 'R']
            self.R_blame[s] = 0.0
        for s in Grid.S:
            if s[0] == 0:
                self.A[s].remove('U')
            if s[0] == init_env.rows - 1:
                self.A[s].remove('D')
            if s[1] == 0:
                self.A[s].remove('L')
            if s[1] == init_env.columns - 1:
                self.A[s].remove('R')
        self.P = 1.0
        self.R = 0.0
        self.gamma = 0.99
        self.V = []
        self.Pi = []
        self.NSE = 0.0
        self.label = label
        self.path = str(self.s)  # + "->"

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

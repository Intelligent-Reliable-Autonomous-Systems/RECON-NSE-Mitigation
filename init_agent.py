import copy
import init_env
from calculation_lib import move

s_goals = init_env.s_goals


class Agent:
    def __init__(self, start_loc, Grid, label):
        # state of an agent: <x,y,trash_size,coral_flag>
        self.s = (start_loc[0], start_loc[1], 'Null', Grid.coral_flag[start_loc[0]][start_loc[1]])
        self.s0 = (start_loc[0], start_loc[1], 'Null', Grid.coral_flag[start_loc[0]][start_loc[1]])
        self.startLoc = start_loc
        self.best_performance_flag = False
        self.goal_states = Grid.goal_states
        self.Grid = Grid

        # operation actions = ['pick_S', 'pick_M', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        self.A = {}
        # counterfactual comparison actions = ['switch_compare']
        self.A2 = {}
        self.R_blame = {}
        for s in Grid.S:
            self.A[s] = ['pick_S', 'pick_M', 'pick_L', 'drop', 'U', 'D', 'L', 'R']  # operation actions
            self.A2[s] = ['switch_compare']  # action for counterfactuals comparison
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

    def Reward(self, s, a, trash_repository):
        # operation actions = ['pick_S', 'pick_M', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,junk_size,coral_flag>
        R = 0
        reward = {'Null': -10, 'S': 10, 'M': 15, 'L': 20}
        # for current implementation: salps cannot drop trash anywhere
        # or pick it up from anywhere but from their own start state
        if a == 'pick_S':
            if s == self.s0:
                if trash_repository['S'] > 0:
                    R = reward['S']
                    trash_repository['S'] -= 1
                else:
                    R = -50
            else:
                R = -50
        elif a == 'pick_M':
            if s == self.s0:
                if trash_repository['M'] > 0:
                    R = reward['M']
                    trash_repository['M'] -= 1
                else:
                    R = -50
            else:
                R = -50
        elif a == 'pick_L':
            if s == self.s0:
                if trash_repository['L'] > 0:
                    R = reward['L']
                    trash_repository['L'] -= 1
                else:
                    R = -50
            else:
                R = -50
        elif a == 'drop':
            if [s[0], s[1]] in self.Grid.goal_states:
                R = 10 * reward[s[2]]
            else:
                R = -100
        else:
            s_ = move(s, a)
            R = self.Grid.R[s_]
            # check this again
        return R, trash_repository

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

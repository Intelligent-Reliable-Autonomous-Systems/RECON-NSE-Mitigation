import copy
import init_env
from calculation_lib import do_action

s_goals = init_env.s_goals
s_goal = init_env.s_goal


class Agent:
    def __init__(self, start_loc, Grid, label):
        self.goal_modes = Grid.goal_modes
        self.label = label
        self.IDX = int(label) - 1
        # state of an agent: <x,y,box_with_me,coral_flag,box_at_goal>
        coral_flag = Grid.coral_flag[start_loc[0]][start_loc[1]]
        self.s = (start_loc[0], start_loc[1], 'X', coral_flag, (0, 0))
        self.s0 = (start_loc[0], start_loc[1], 'X', coral_flag, self.goal_modes[self.IDX])
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
        self.path = str(self.s)  # + "->"
        self.plan = ""
        self.trajectory = []
        self.num_of_agents = Grid.num_of_agents
        self.done_flag = False

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
            if s[2] != 'X':
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if s[2] == 'X':
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
            if s[4][0] >= s_goal[4][0]:
                if 'pick_S' in self.A[s]:
                    self.A[s].remove('pick_S')
            if s[4][1] >= s_goal[4][1]:
                if 'pick_L' in self.A[s]:
                    self.A[s].remove('pick_L')
            if s[2] == 'S' and s[4][0] >= s_goal[4][0]:
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
            if s[2] == 'L' and s[4][1] >= s_goal[4][1]:
                if 'drop' in self.A[s]:
                    self.A[s].remove('drop')
        for s in Grid.S:
            print("A[" + str(s) + "] = " + str(self.A[s]))
        # for s in Grid.S:
        #     if len(s[4]) == 2 and s[2] != 'X':
        #         del Grid.S[s]
        #     if len(s[4]) == 2:
        #         if 'pick_S' in self.A[s]:
        #             self.A[s].remove('pick_S')
        #         if 'pick_L' in self.A[s]:
        #             self.A[s].remove('pick_L')

    def Reward(self, s, a):
        # operation actions = ['pick_S', 'pick_L', 'drop', 'U', 'D', 'L', 'R']
        # state of an agent: <x,y,trash_box_size,coral_flag,list_of_boxes_at_goal>
        if init_env.do_action(self, s, a) == s_goal:
            R = 100
        else:
            R = -1
        return R

    def follow_policy(self):
        Pi = copy.copy(self.Pi)

        while self.s != s_goal:
            R = self.Reward(self.s, Pi[self.s])
            self.R += R
            self.trajectory.append((self.s, Pi[self.s], R))
            self.plan += " -> " + str(Pi[self.s])
            # print("Action: ", Pi[self.s])
            # print("Plan till here: ", self.plan)
            self.s = do_action(self, self.s, Pi[self.s])
            x, y, _, _, _ = self.s
            loc = (x, y)
            self.path = self.path + "->" + str(loc)

    def agent_reset(self):
        self.NSE = 0.0
        self.s = copy.deepcopy(self.s0)
        self.path = str(self.s)  # + "->"
        self.R = 0.0

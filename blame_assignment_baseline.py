import copy
import numpy as np
from init_env import log_joint_NSE
from itertools import permutations


class BlameBaseline:
    def __init__(self, Agents, Grid):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0  # best NSE is no NSE
        self.NSE_worst = Grid.max_log_joint_NSE(Agents)
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.01

    def get_blame(self, original_NSE, joint_NSE_state):
        """
        :param original_NSE: Scalar value of NSE from a single joint state "joint_NSE_state"
        :param joint_NSE_state: the joint state under investigation
        :return: numpy 1d array of individual agent blames
        """
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))

        for agent_idx in range(len(self.Agents)):
            counterfactual_constant_state = copy.deepcopy(joint_NSE_state)
            counterfactual_constant_state = list(counterfactual_constant_state)
            counterfactual_constant_state[agent_idx] = list(counterfactual_constant_state[agent_idx])
            counterfactual_constant_state[agent_idx][2] = 'B'  # replacing i^th agents state with constant cf state
            counterfactual_constant_state[agent_idx] = tuple(counterfactual_constant_state[agent_idx])
            counterfactual_constant_state = tuple(counterfactual_constant_state)
            baseline_performance_by_agent = self.Grid.give_joint_NSE_value(counterfactual_constant_state)

            if original_NSE <= baseline_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False

            blame_val = round(original_NSE - baseline_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon  # + self.NSE_worst
            # blame[agent_idx] = round(blame[agent_idx] / 2, 2)
            # if joint_NSE_state[agent_idx][2] == 'X':
            #     blame[agent_idx] = 0.0
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = blame[agent_idx]  # * original_NSE / np.sum(blame[:])
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        # factor = original_NSE / np.sum(blame[:])
        # print(str(NSE_blame) + ": " + str(sum(NSE_blame)))
        # print("Original NSE: ", original_NSE)
        # print("Factor: ", factor)
        # NSE_Blame = [factor * i for i in NSE_blame]

        # NSE_Blame = np.zeros(len(self.Agents))
        return NSE_blame

    def get_training_data_with_cf(self, Agents, Joint_NSE_states):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            _, all_cfs_for_this_js = generate_counterfactuals(js, self.Agents)
            for cf_state_in_all_cfs_for_js in all_cfs_for_this_js:
                if cf_state_in_all_cfs_for_js not in joint_NSE_states:
                    joint_NSE_states = joint_NSE_states + [cf_state_in_all_cfs_for_js]
        for js in joint_NSE_states:
            original_NSE = self.Grid.give_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                agent.blame_training_data_x_with_cf.append([weighting[s[2]], int(s[3]), s[4][0], s[4][1]])
                agent.blame_training_data_y_with_cf.append(-blame_values[agent_idx])

    def get_training_data_wo_cf(self, Agents, Joint_NSE_states):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            original_NSE = self.Grid.give_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                agent.blame_training_data_x_wo_cf.append([weighting[s[2]], int(s[3]), s[4][0], s[4][1]])
                agent.blame_training_data_y_wo_cf.append(-blame_values[agent_idx])

    def compute_R_Blame_for_all_Agents(self, Agents, joint_NSE_states):
        blame_distribution = {}  # blame distributions of joint states [Agent1_blame, Agent2_blame,..]

        for js_nse in joint_NSE_states:
            original_NSE = self.Grid.give_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)

        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_dr[s] = blame_array_for_js[agent.IDX]
                self.Grid.add_goal_reward(agent)

    def compute_considerate_R_Blame_for_all_Agents(self, Agents, joint_NSE_states):
        blame_distribution = {}  # blame distributions of joint states [Agent1_blame, Agent2_blame,..]
        alpha_self = 1.0
        alpha_care = 0.2


        for js_nse in joint_NSE_states:
            original_NSE = self.Grid.give_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)

        for js_nse in joint_NSE_states:
            original_NSE = - self.Grid.give_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent
            for agent in Agents:
                blame_array_for_js = - blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = alpha_self * blame_array_for_js[agent.IDX] +  alpha_care * (original_NSE - blame_array_for_js[agent.IDX])

def generate_counterfactuals(joint_state, Agents):
    # permuting s[2] from state s: <x,y,junk_size,coral_flag,goal_trash> in joint state (s1,s2,s3...)
    counterfactual_jointStates = []
    agent_wise_cfStates = []
    all_cf_joint_states = []
    for agent in Agents:
        cf_joint_state = []
        Joint_State = copy.deepcopy(joint_state)
        agent_idx = int(agent.label) - 1
        size_options = ['A', 'B']
        # print('BEFORE size options for Agent ' + agent.label + ' cfs: ' + str(size_options))
        # print(Joint_State[agent_idx][2])
        if Joint_State[agent_idx][2] in size_options:
            size_options.remove(Joint_State[agent_idx][2])  # agent should choose something different as counterfactual
        if Joint_State[agent_idx][2] == 'X':
            size_options = ['X']

        # print('AFTER size options for Agent ' + agent.label + ' cfs: ' + str(size_options))
        for option in size_options:
            s = Joint_State[agent_idx]
            s = list(s)
            s[2] = option
            s = tuple(s)
            Joint_State = list(Joint_State)
            Joint_State[agent_idx] = s
            Joint_State = tuple(Joint_State)
            cf_joint_state.append(Joint_State)
            counterfactual_jointStates.append(Joint_State)
            all_cf_joint_states.append(Joint_State)
        agent_wise_cfStates.append(counterfactual_jointStates)
        counterfactual_jointStates = []

    return agent_wise_cfStates, all_cf_joint_states

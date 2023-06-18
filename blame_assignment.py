import copy
import numpy as np
from init_env import log_joint_NSE, weighting
from itertools import permutations


class Blame:
    def __init__(self, Agents, Grid):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0  # best NSE is no NSE
        self.NSE_worst = Grid.max_log_joint_NSE(Agents)
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.001

    def assign_blame_using_clone(self, joint_state):
        # clone agent 1 by deep-copying it
        clone_agent = copy.deepcopy(self.Agents[0])
        clone_agent.s = joint_state[0]
        Agents = copy.deepcopy(self.Agents)
        index = 0
        for agent in Agents:
            agent.s = joint_state[index]
            index += 1
        Agents_augmented = copy.deepcopy(Agents)
        Agents_augmented.append(
            clone_agent)  # this is not a list of all agents at the joint state under investigation + clone agent1
        NSE_original = self.get_NSE(Agents)
        NSE_with_clone = self.get_NSE(Agents_augmented)
        b1 = NSE_with_clone - NSE_original
        b2 = 2 * NSE_original - NSE_with_clone
        self.blame = np.array([b1, b2])
        return [b1, b2]

    def get_NSE(self, Agents):
        NSE_val = 0
        for agent in Agents:
            NSE_val += self.Grid.NSE[agent.s]
        return NSE_val

    def get_blame(self, original_NSE, joint_NSE_state):
        """
        :param original_NSE: Scalar value of NSE from a single joint state "joint_NSE_state"
        :param joint_NSE_state: the joint state under investigation
        :return: numpy 1d array of individual agent blames
        """
        agentWise_cfs_NSEs = []
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))
        # print("~~~~~~     Original NSE: ", original_NSE)
        # print("joint_NSE_state in the function(line 47): ", joint_NSE_state)
        # print("~~~~~~     Original NSE: ", get_joint_NSEs_for_list(joint_NSE_state))
        agentWise_cfs, _ = generate_counterfactuals(joint_NSE_state, self.Agents)
        for cf_state in agentWise_cfs:
            # print('[blame_assignment] cf_state: ', cf_state)
            NSEs_for_cf_state = get_joint_NSEs_for_list(cf_state, self.Grid)
            # print("[blame_assignment] ****cf_NSEs for agent: " + str(NSEs_for_cf_state))
            agentWise_cfs_NSEs.append(NSEs_for_cf_state)
        # print('[blame_assignment] agentWise_cfs_NSEs: ', agentWise_cfs_NSEs)
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        for agent_idx in range(len(self.Agents)):
            cf_nse_set_for_agent = agentWise_cfs_NSEs[agent_idx]
            # print("[blame_assignment] cf_nse_set_for_agent: ", cf_nse_set_for_agent)
            # print("min(cf_nse_set_for_agent): ", min(cf_nse_set_for_agent))
            best_performance_by_agent = min(list(cf_nse_set_for_agent))

            if original_NSE <= best_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False

            blame_val = round(original_NSE - best_performance_by_agent, 2)  # the worst blame can be 0
            # print('[blame_assignment]          blame_val: ', blame_val)
            # print("min(CF_NSEs Agent" + str(agent_idx + 1) + ") -> " + str(best_performance_by_agent))
            # print("Blame = OG_NSE - min(CF_NSEs Agent" + str(agent_idx + 1) + ") -> ", blame_val)
            blame[agent_idx] = blame_val + self.epsilon + self.NSE_worst
            blame[agent_idx] = round(blame[agent_idx] / 2, 2)  # rescale Blame from [0, 2*NSE_worst] to [0,NSE_worst]
            if joint_NSE_state[agent_idx][2] == 'X':
                blame[agent_idx] = 0.0
            # print("---- AFTER Blame Value =", blame[agent_idx])
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = round((((blame[agent_idx]) / (np.sum(blame[:]))) * original_NSE), 2)
        # for agent in self.Agents:
        # if agent.best_performance_flag is True:
        # print("Agent " + agent.label + " did it's best!")
        # else:
        # print("Agent " + agent.label + " did NOT do it's best!")
        return NSE_blame

    def get_training_data(self, Agents, Joint_NSE_states):
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
                agent.blame_training_data_x.append([weighting[s[2]], int(s[3]), s[4][0], s[4][1]])
                agent.blame_training_data_y.append(-blame_values[agent_idx])


def generate_counterfactuals(joint_state, Agents):
    # permuting s[2] from state s: <x,y,junk_size,coral_flag,goal_trash> in joint state (s1,s2,s3...)
    counterfactual_jointStates = []
    agent_wise_cfStates = []
    all_cf_joint_states = []
    for agent in Agents:
        cf_joint_state = []
        Joint_State = copy.deepcopy(joint_state)
        agent_idx = int(agent.label) - 1
        size_options = ['S', 'L']
        # print('BEFORE size options for Agent ' + agent.label + ' cfs: ' + str(size_options))
        # print(Joint_State[agent_idx][2])
        if Joint_State[agent_idx][2] in size_options:
            size_options.remove(Joint_State[agent_idx][2])  # agent should choose something different as counterfactual

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


def get_joint_NSEs_for_list(joint_states, Grid):
    joint_NSEs = []
    for js in joint_states:
        nse = Grid.give_joint_NSE_value(js)
        joint_NSEs.append(nse)
    return joint_NSEs

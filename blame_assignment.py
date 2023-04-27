import copy
import numpy as np
from init_env import log_joint_NSE
from itertools import permutations


class Blame:
    def __init__(self, Agents, Grid):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0  # best NSE is no NSE
        self.NSE_worst = Grid.max_log_joint_NSE(Agents)
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.0001

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
        agentWise_cfs_NSEs = []
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))
        # print("~~~~~~     Original NSE: ", original_NSE)
        # print("joint_NSE_state in the function(line 47): ", joint_NSE_state)
        # print("~~~~~~     Original NSE: ", get_joint_NSEs_for_list(joint_NSE_state))
        agentWise_cfs, cfs = generate_counterfactuals(joint_NSE_state, self.Grid, self.Agents, print_flag=False)
        for cf_state in agentWise_cfs:
            NSEs_for_cf_state = get_joint_NSEs_for_list(cf_state, self.Grid)
            # print("****cf_NSEs for agent: " + str(cf_NSEs_for_agent_i))
            agentWise_cfs_NSEs.append(NSEs_for_cf_state)
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        for agent_idx in range(len(self.Agents)):
            cf_nse_set_for_agent = agentWise_cfs_NSEs[agent_idx]
            # print("cf_nse_set_for_agent: ", cf_nse_set_for_agent)
            # print("min(cf_nse_set_for_agent): ", min(cf_nse_set_for_agent))
            best_performance_by_agent = min(list(cf_nse_set_for_agent))

            if original_NSE <= best_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False

            blame_val = round(original_NSE - best_performance_by_agent, 2)  # the worst blame can be 0
            # print("min(CF_NSEs Agent" + str(agent_idx + 1) + ") -> " + str(best_performance_by_agent))
            # print("Blame = OG_NSE - min(CF_NSEs Agent" + str(agent_idx + 1) + ") -> ", blame_val)
            blame[agent_idx] = blame_val + self.NSE_worst + self.epsilon
            blame[agent_idx] = round(blame[agent_idx] / 2, 2)  # rescale Blame from [0, 2*NSE_worst] to [0,NSE_worst]
            # print("---- AFTER Blame Value =", blame[agent_idx])
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = round((((blame[agent_idx]) / (np.sum(blame[:]) + 0.001)) * original_NSE), 2)
        for agent in self.Agents:
            if agent.best_performance_flag is True:
                print("Agent " + agent.label + " did it's best!")
            else:
                print("Agent " + agent.label + " did NOT do it's best!")
        return NSE_blame


def generate_counterfactuals(joint_state, Grid, Agents, print_flag=True):
    # permuting s[2] from state s: <x,y,junk_size,coral_flag> in joint state (s1,s2,s3...)
    counterfactual_jointStates = []
    agent_wise_cfStates = []
    for agent in Agents:
        cf_joint_state = []
        Joint_State = copy.deepcopy(joint_state)
        agent_idx = int(agent.label) - 1
        size_options = list(Grid.trash_repository.keys())
        for junk_unit_size in list(Grid.trash_repository.keys()):
            if Grid.trash_repository[junk_unit_size] <= 0:
                size_options.remove(junk_unit_size)

        if Grid.trash_repository[agent.s[2]] != 0:
            size_options.remove(agent.s[2])  # we want agent to choose something different as a counterfactual

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
        agent_wise_cfStates.append(counterfactual_jointStates)
        counterfactual_jointStates = []

    return agent_wise_cfStates, counterfactual_jointStates


def get_joint_NSEs_for_list(joint_states, Grid):
    joint_NSEs = []
    for js in joint_states:
        nse = Grid.give_joint_NSE_value(js)
        joint_NSEs.append(nse)
    return joint_NSEs

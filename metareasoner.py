import copy
import numpy as np
from typing import List
from salp_mdp import SalpAgent, SalpEnvironment
from overcooked_mdp import OvercookedAgent, OvercookedEnvironment
from warehouse_mdp import WarehouseAgent, WarehouseEnvironment

############################################################################################################
############################################################################################################
########################################  SALP METAREASONER  ###############################################
############################################################################################################

class SalpMetareasoner:
    def __init__(self, Agents:List[SalpAgent], Grid:SalpEnvironment):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0 
        self.NSE_worst = self.max_log_joint_NSE()
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.01

    def get_blame(self, original_NSE:float, joint_NSE_state:float):
        """
        :param original_NSE: Scalar value of NSE from a single joint state "joint_NSE_state"
        :param joint_NSE_state: the joint state under investigation
        :return: numpy 1d array of individual agent blames
        """
        agentWise_cfs_NSEs = []
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))
        agentWise_cfs, _ = self.generate_counterfactuals(joint_NSE_state, self.Agents)
        for cf_state in agentWise_cfs:
            NSEs_for_cf_state = self.get_joint_NSEs_for_list(cf_state)
            agentWise_cfs_NSEs.append(NSEs_for_cf_state)
        for agent_idx in range(len(self.Agents)):
            cf_nse_set_for_agent = agentWise_cfs_NSEs[agent_idx]
            best_performance_by_agent = min(list(cf_nse_set_for_agent))
            if original_NSE <= best_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - best_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon + self.NSE_worst
            blame[agent_idx] = round(blame[agent_idx] / 2, 2)
            if joint_NSE_state[agent_idx][2] == 'X':
                blame[agent_idx] = 0.0
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = round((((blame[agent_idx]) / (np.sum(blame[:]))) * original_NSE), 2)
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_for_all_Agents(self, Agents:List[SalpAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame[s] = blame_array_for_js[agent.IDX]

    def compute_considerate_reward_for_all_Agents(self, Agents:List[SalpAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent (+ve value)
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]  # (-ve value)
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = (-original_NSE - blame_array_for_js[agent.IDX])/agent.Grid.R_Nmax  # (-ve value)

    def generate_counterfactuals(self, joint_state, Agents:List[SalpAgent]):
        counterfactual_jointStates = []
        agent_wise_cfStates = []
        all_cf_joint_states = []
        for agent in Agents:
            cf_joint_state = []
            Joint_State = copy.deepcopy(joint_state)
            agent_idx = int(agent.label) - 1
            size_options = ['A', 'B']
            if Joint_State[agent_idx][2] in size_options:
                size_options.remove(Joint_State[agent_idx][2])  # agent should choose something different as counterfactual
            if Joint_State[agent_idx][2] == 'X':
                size_options = ['X']
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

    def get_training_data_with_cf(self, Agents:List[SalpAgent], Joint_NSE_states:List[tuple]):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            _, all_cfs_for_this_js = self.generate_counterfactuals(js, self.Agents)
            for cf_state_in_all_cfs_for_js in all_cfs_for_this_js:
                if cf_state_in_all_cfs_for_js not in joint_NSE_states:
                    joint_NSE_states = joint_NSE_states + [cf_state_in_all_cfs_for_js]
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                agent.blame_training_data_x_with_cf.append(np.array([weighting[s[2]], int(s[3])]))
                agent.blame_training_data_y_with_cf.append(-blame_values[agent_idx])

    def get_training_data_wo_cf(self, Agents:List[SalpAgent], Joint_NSE_states:List[tuple]):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                # s = (s[0]: i,s[1]: j,s[2]: sample, s[3]: coral_flag ,s[4]: done)
                agent.blame_training_data_x_wo_cf.append(np.array([weighting[s[2]], int(s[3])]))
                agent.blame_training_data_y_wo_cf.append(-blame_values[agent_idx])

    def get_total_blame_breakdown(self, joint_NSE_states:List):
        blame_distribution = []
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution.append(blame_values)
        # sum all the blame values for each agent
        total_blame = np.sum(blame_distribution, axis=0)
        return total_blame
        
    
    def get_joint_NSEs_for_list(self, joint_states):
        joint_NSEs = []
        for js in joint_states:
            nse = self.get_joint_NSE_value(js)
            joint_NSEs.append(nse)
        return joint_NSEs

        
    def get_joint_NSE_value(self, joint_state):
        joint_NSE_val = self.log_joint_NSE(joint_state)
        return joint_NSE_val
    
    def log_joint_NSE(self, joint_state):
        joint_NSE_val = 0
        X = {}
        sample_at_coral_loc = {}
        coral_locs = np.argwhere(self.Grid.All_States == 'C')
        for coral_loc in coral_locs:
            X[coral_loc[0], coral_loc[1]] = 0
            sample_at_coral_loc[coral_loc[0], coral_loc[1]] = 'X'
        # count the number of agents on the same coral location with a sample
        for s in joint_state:
            if s[3] is True and s[2] != 'X':
                X[s[0], s[1]] += 1
                sample_at_coral_loc[s[0], s[1]] = s[2]
        # calculate the log of the number of agents on the same coral location multiplied with the weighting of the sample
        for coral_loc in coral_locs:
            joint_NSE_val += self.Grid.weighting[sample_at_coral_loc[coral_loc[0], coral_loc[1]]] * np.log(X[coral_loc[0], coral_loc[1]] + 1)
        # rescale the joint_NSE_val to get good values
        joint_NSE_val *= 1
        joint_NSE_val = round(joint_NSE_val, 2)
        return joint_NSE_val
        
    def max_log_joint_NSE(self):
        NSE_worst = max(self.Grid.weighting.values()) * np.log(self.Grid.num_of_agents + 1)
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst
    
    def get_jointstates_and_NSE_list(self, Agents:List[SalpAgent], report_name="No Print"):
        '''
        :param Agents: List of agents
        :param report_name: Name of the report to be printed if at all'''
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        path_joint_states = [self.Grid.get_joint_state(Agents)]  # Store the starting joint states
        path_joint_NSE_values = [self.get_joint_NSE_value(self.Grid.get_joint_state(Agents))]  # Store the corresponding joint NSE
        joint_NSE_states = []
        joint_NSE_values = []
        while self.Grid.all_have_reached_goal(Agents) is False:
            for agent in Agents:
                agent.s = agent.step(agent.s, agent.Pi[agent.s])
            next_joint_state = self.Grid.get_joint_state(Agents)
            joint_NSE = self.get_joint_NSE_value(next_joint_state)
            joint_state = next_joint_state
            path_joint_states.append(joint_state)
            path_joint_NSE_values.append(joint_NSE)
            joint_NSE_states.append(joint_state)
            joint_NSE_values.append(joint_NSE)
        if report_name != "No Print":
            print(report_name+ ' policy report: ')
            for x in range(len(path_joint_states)):
                print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))
        return joint_NSE_states, path_joint_NSE_values
    
    def get_total_R_and_NSE_from_path(self, Agents:List[SalpAgent], path_joint_NSE_values):
        R = 0  # Just for storage purposes
        NSE = 0  # Just for storage purposes
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        R = [round(agent.R, 2) for agent in Agents]
        NSE = round(float(np.sum(path_joint_NSE_values)), 2)
        print("Total Reward: ", sum(R))
        print("Total NSE: ", NSE)
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        return R, NSE
    
    def get_blame_DR(self, original_NSE, joint_NSE_state):
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
            counterfactual_constant_state[agent_idx][2] = 'B'  
            counterfactual_constant_state[agent_idx] = tuple(counterfactual_constant_state[agent_idx])
            counterfactual_constant_state = tuple(counterfactual_constant_state)
            baseline_performance_by_agent = self.get_joint_NSE_value(counterfactual_constant_state)
            if original_NSE <= baseline_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - baseline_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon  
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = blame[agent_idx] 
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_dr_for_all_Agents(self, Agents:List[SalpAgent], joint_NSE_states):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame_DR(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_dr[s] = blame_array_for_js[agent.IDX]
                
    def compute_considerate_reward_for_all_Agents(self, Agents:List[SalpAgent], joint_NSE_states):
        blame_distribution = {}  
        alpha_self = 0.5
        alpha_care = 0.5
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = - self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent
            for agent in Agents:
                blame_array_for_js = - blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = alpha_self * blame_array_for_js[agent.IDX] +  alpha_care * (original_NSE - blame_array_for_js[agent.IDX])

############################################################################################################
############################################################################################################
########################################  OVERCOOKED METAREASONER  #########################################
############################################################################################################
############################################################################################################

class OvercookedMetareasoner:
    def __init__(self, Agents:List[OvercookedAgent], Grid:OvercookedEnvironment):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0  # best NSE is no NSE
        self.NSE_worst = self.max_log_joint_NSE()
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.01

    def get_blame(self, original_NSE:float, joint_NSE_state:float):
        """
        :param original_NSE: Scalar value of NSE from a single joint state "joint_NSE_state"
        :param joint_NSE_state: the joint state under investigation
        :return: numpy 1d array of individual agent blames
        """
        # overcooked state = < s[0]: i,s[1]: j,s[2]: dir,s[3]: dustbin ,s[4]: in_hand_object,s[5]: done>
        agentWise_cfs_NSEs = []
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))
        agentWise_cfs, _ = self.generate_counterfactuals(joint_NSE_state, self.Agents)
        for cf_state in agentWise_cfs:
            NSEs_for_cf_state = self.get_joint_NSEs_for_list(cf_state)
            agentWise_cfs_NSEs.append(NSEs_for_cf_state)
        for agent_idx in range(len(self.Agents)):
            cf_nse_set_for_agent = agentWise_cfs_NSEs[agent_idx]
            best_performance_by_agent = min(list(cf_nse_set_for_agent))
            if original_NSE <= best_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - best_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon + self.NSE_worst
            blame[agent_idx] = round(blame[agent_idx] / 2, 2)
            if joint_NSE_state[agent_idx][4] == 'X':
                blame[agent_idx] = 0.0
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = round((((blame[agent_idx]) / (np.sum(blame[:]))) * original_NSE), 2)
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_for_all_Agents(self, Agents:List[OvercookedAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {} 
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame[s] = blame_array_for_js[agent.IDX]

    def compute_considerate_reward_for_all_Agents(self, Agents:List[OvercookedAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent (+ve value)
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]  # (-ve value)
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = (-original_NSE - blame_array_for_js[agent.IDX])/agent.Grid.R_Nmax  # (-ve value)

    def generate_counterfactuals(self, joint_state, Agents:List[OvercookedAgent]):
        counterfactual_jointStates = []
        agent_wise_cfStates = []
        all_cf_joint_states = []
        for agent in Agents:
            cf_joint_state = []
            Joint_State = copy.deepcopy(joint_state)
            agent_idx = int(agent.label) - 1
            size_options = ['D', 'T', 'O', 'Dt', 'Do']
            if Joint_State[agent_idx][2] in size_options:
                size_options.remove(Joint_State[agent_idx][2])  # agent should choose something different as counterfactual
            if Joint_State[agent_idx][2] == 'X':
                size_options = ['X']
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

    def get_training_data_with_cf(self, Agents:List[OvercookedAgent], Joint_NSE_states:List[tuple]):
        # overcooked state = < s[0]: i,s[1]: j,s[2]: dir,s[3]: dustbin ,s[4]: in_hand_object,s[5]: done>
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            _, all_cfs_for_this_js = self.generate_counterfactuals(js, self.Agents)
            for cf_state_in_all_cfs_for_js in all_cfs_for_this_js:
                if cf_state_in_all_cfs_for_js not in joint_NSE_states:
                    joint_NSE_states = joint_NSE_states + [cf_state_in_all_cfs_for_js]
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                agent.blame_training_data_x_with_cf.append(np.array([int(s[3]), weighting[s[4]]]))
                agent.blame_training_data_y_with_cf.append(-blame_values[agent_idx])

    def get_training_data_wo_cf(self, Agents:List[OvercookedAgent], Joint_NSE_states:List[tuple]):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                # overcooked state = < s[0]: i,s[1]: j,s[2]: dir,s[3]: dustbin ,s[4]: in_hand_object,s[5]: done>
                agent.blame_training_data_x_wo_cf.append(np.array([int(s[3]), weighting[s[4]]]))
                agent.blame_training_data_y_wo_cf.append(-blame_values[agent_idx])

    def get_total_blame_breakdown(self, joint_NSE_states:List):
        blame_distribution = []
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution.append(blame_values)
        # sum all the blame values for each agent
        total_blame = np.sum(blame_distribution, axis=0)
        return total_blame
    
    def get_joint_NSEs_for_list(self, joint_states):
        joint_NSEs = []
        for js in joint_states:
            nse = self.get_joint_NSE_value(js)
            joint_NSEs.append(nse)
        return joint_NSEs
        
    def get_joint_NSE_value(self, joint_state):
        joint_NSE_val = self.log_joint_NSE(joint_state)
        return joint_NSE_val
    
    def log_joint_NSE(self, joint_state):
        joint_NSE_val = 0
        X = {}
        for item_type in self.Grid.weighting.keys():
            X[item_type] = 0
        Joint_State = list(copy.deepcopy(joint_state))
        # s = < s[0]: i,s[1]: j,s[2]: dir,s[3]: dustbin ,s[4]: in_hand_object,s[5]: done>
        for s in Joint_State:
            if s[3] is True:
                X[s[4]] += 1
        for item_type in X.keys():
            joint_NSE_val += self.Grid.weighting[item_type] * np.log(X[item_type] + 1)
        # joint_NSE_val *= 1  # rescaling it to get good values
        joint_NSE_val = round(joint_NSE_val, 2)
        return joint_NSE_val
        
    def max_log_joint_NSE(self):
        NSE_worst = max(self.Grid.weighting.values()) * np.log(self.Grid.num_of_agents + 1)
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst
    
    def get_jointstates_and_NSE_list(self, Agents:List[OvercookedAgent], report_name="No Print"):
        '''
        :param Agents: List of agents
        :param report_name: Name of the report to be printed if at all'''
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        path_joint_states = [self.Grid.get_joint_state(Agents)]  # Store the starting joint states
        path_joint_NSE_values = [self.get_joint_NSE_value(self.Grid.get_joint_state(Agents))]  # Store the corresponding joint NSE
        joint_NSE_states = []
        joint_NSE_values = []
        while self.Grid.all_have_reached_goal(Agents) is False:
            for agent in Agents:
                agent.s = agent.step(agent.s, agent.Pi[agent.s])
            next_joint_state = self.Grid.get_joint_state(Agents)
            joint_NSE = self.get_joint_NSE_value(next_joint_state)
            joint_state = next_joint_state
            path_joint_states.append(joint_state)
            path_joint_NSE_values.append(joint_NSE)
            joint_NSE_states.append(joint_state)
            joint_NSE_values.append(joint_NSE)
        if report_name != "No Print":
            print(report_name+ ' policy report: ')
            for x in range(len(path_joint_states)):
                print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))
        return joint_NSE_states, path_joint_NSE_values
    
    def get_total_R_and_NSE_from_path(self, Agents:List[OvercookedAgent], path_joint_NSE_values):
        R = 0 
        NSE = 0 
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        R = [round(agent.R, 2) for agent in Agents]
        NSE = round(float(np.sum(path_joint_NSE_values)), 2)
        print("Total Reward: ", sum(R))
        print("Total NSE: ", NSE)
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        return R, NSE
    
    def get_blame_DR(self, original_NSE, joint_NSE_state):
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
            counterfactual_constant_state[agent_idx][2] = max(self.Grid.weighting, key=self.Grid.weighting.get)  # most nse penalty item  
            counterfactual_constant_state[agent_idx] = tuple(counterfactual_constant_state[agent_idx])
            counterfactual_constant_state = tuple(counterfactual_constant_state)
            baseline_performance_by_agent = self.get_joint_NSE_value(counterfactual_constant_state)
            if original_NSE <= baseline_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - baseline_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon  
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = blame[agent_idx] 
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_dr_for_all_Agents(self, Agents:List[OvercookedAgent], joint_NSE_states):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame_DR(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_dr[s] = blame_array_for_js[agent.IDX]
                # self.Grid.add_goal_reward(agent)

    def compute_considerate_reward_for_all_Agents(self, Agents:List[OvercookedAgent], joint_NSE_states):
        blame_distribution = {}  
        alpha_self = 0.5
        alpha_care = 0.5
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = - self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent
            for agent in Agents:
                blame_array_for_js = - blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = alpha_self * blame_array_for_js[agent.IDX] +  alpha_care * (original_NSE - blame_array_for_js[agent.IDX])


############################################################################################################
############################################################################################################
#########################################  WAREHOUSE METAREASONER  #########################################
############################################################################################################
############################################################################################################

class WarehouseMetareasoner:
    def __init__(self, Agents:List[WarehouseAgent], Grid:WarehouseEnvironment):
        self.blame = np.array(len(Agents))
        self.Agents = Agents
        self.Grid = Grid
        self.NSE_best = 0  # best NSE is no NSE
        self.NSE_worst = self.max_log_joint_NSE()
        # print("Worst NSE: ", self.NSE_worst)
        self.NSE_window = (self.NSE_best, self.NSE_worst)  # formulation where NSE is positive
        self.epsilon = 0.01

    def get_blame(self, original_NSE:float, joint_NSE_state:float):
        """
        :param original_NSE: Scalar value of NSE from a single joint state "joint_NSE_state"
        :param joint_NSE_state: the joint state under investigation
        :return: numpy 1d array of individual agent blames
        """
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        agentWise_cfs_NSEs = []
        blame = np.zeros(len(self.Agents))
        NSE_blame = np.zeros(len(self.Agents))
        agentWise_cfs, _ = self.generate_counterfactuals(joint_NSE_state, self.Agents)
        for cf_state in agentWise_cfs:
            NSEs_for_cf_state = self.get_joint_NSEs_for_list(cf_state)
            agentWise_cfs_NSEs.append(NSEs_for_cf_state)
        for agent_idx in range(len(self.Agents)):
            cf_nse_set_for_agent = agentWise_cfs_NSEs[agent_idx]
            best_performance_by_agent = min(list(cf_nse_set_for_agent))
            if original_NSE <= best_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - best_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon + self.NSE_worst
            blame[agent_idx] = round(blame[agent_idx] / 2, 2)
            if joint_NSE_state[agent_idx][4] == 'X':
                blame[agent_idx] = 0.0
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = round((((blame[agent_idx]) / (np.sum(blame[:]))) * original_NSE), 2)
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_for_all_Agents(self, Agents:List[WarehouseAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame[s] = blame_array_for_js[agent.IDX]

    def compute_considerate_reward_for_all_Agents(self, Agents:List[WarehouseAgent], joint_NSE_states:List[tuple]):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent (+ve value)
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]  # (-ve value)
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = (-original_NSE - blame_array_for_js[agent.IDX])/agent.Grid.R_Nmax  # (-ve value)

    def generate_counterfactuals(self, joint_state, Agents:List[WarehouseAgent]):
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        counterfactual_jointStates = []
        agent_wise_cfStates = []
        all_cf_joint_states = []
        for agent in Agents:
            cf_joint_state = []
            Joint_State = copy.deepcopy(joint_state)
            agent_idx = int(agent.label) - 1
            size_options = ['s1', 's2', 'S1', 'S2']
            if Joint_State[agent_idx][2] in size_options:
                size_options.remove(Joint_State[agent_idx][2])  # agent should choose something different as counterfactual
            if Joint_State[agent_idx][2] == 'X':
                size_options = ['X']
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

    def get_training_data_with_cf(self, Agents:List[WarehouseAgent], Joint_NSE_states:List[tuple]):
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            _, all_cfs_for_this_js = self.generate_counterfactuals(js, self.Agents)
            for cf_state_in_all_cfs_for_js in all_cfs_for_this_js:
                if cf_state_in_all_cfs_for_js not in joint_NSE_states:
                    joint_NSE_states = joint_NSE_states + [cf_state_in_all_cfs_for_js]
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                agent.blame_training_data_x_with_cf.append(np.array([weighting[s[2]], int(s[3])]))
                agent.blame_training_data_y_with_cf.append(-blame_values[agent_idx])

    def get_training_data_wo_cf(self, Agents:List[WarehouseAgent], Joint_NSE_states:List[tuple]):
        weighting = copy.deepcopy(self.Grid.weighting)
        joint_NSE_states = copy.deepcopy(Joint_NSE_states)
        for js in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js)
            blame_values = self.get_blame(original_NSE, js)
            blame_values = np.around(blame_values, 2)
            for agent in Agents:
                agent_idx = agent.IDX
                s = js[agent_idx]
                # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
                agent.blame_training_data_x_wo_cf.append(np.array([weighting[s[2]], int(s[3])]))
                agent.blame_training_data_y_wo_cf.append(-blame_values[agent_idx])

    def get_total_blame_breakdown(self, joint_NSE_states:List):
        blame_distribution = []
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution.append(blame_values)
        # sum all the blame values for each agent
        total_blame = np.sum(blame_distribution, axis=0)
        return total_blame
    
    def get_joint_NSEs_for_list(self, joint_states):
        joint_NSEs = []
        for js in joint_states:
            nse = self.get_joint_NSE_value(js)
            joint_NSEs.append(nse)
        return joint_NSEs

        
    def get_joint_NSE_value(self, joint_state):
        joint_NSE_val = self.log_joint_NSE(joint_state)
        return joint_NSE_val
    
    def log_joint_NSE(self, joint_state):
        joint_NSE_val = 0
        X = {}
        for item_type in self.Grid.weighting.keys():
            X[item_type] = 0
        Joint_State = list(copy.deepcopy(joint_state))
        # warehouse s = (s[0]: i, s[1]: j, s[2]: shelf, s[3]: narrow_corridor, s[4]: done)
        for s in Joint_State:
            if s[3] is True:
                X[s[2]] += 1
        for item_type in X.keys():
            joint_NSE_val += self.Grid.weighting[item_type] * np.log(X[item_type] + 1)
        joint_NSE_val = round(joint_NSE_val, 2)
        return joint_NSE_val
    
    def max_log_joint_NSE(self):
        NSE_worst = max(self.Grid.weighting.values()) * np.log(self.Grid.num_of_agents + 1)
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst
    
    def get_jointstates_and_NSE_list(self, Agents:List[WarehouseAgent], report_name="No Print"):
        '''
        :param Agents: List of agents
        :param report_name: Name of the report to be printed if at all'''
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        path_joint_states = [self.Grid.get_joint_state(Agents)]  # Store the starting joint states
        path_joint_NSE_values = [self.get_joint_NSE_value(self.Grid.get_joint_state(Agents))]  # Store the corresponding joint NSE
        joint_NSE_states = []
        joint_NSE_values = []
        while self.Grid.all_have_reached_goal(Agents) is False:
            for agent in Agents:
                agent.s = agent.step(agent.s, agent.Pi[agent.s])
            next_joint_state = self.Grid.get_joint_state(Agents)
            joint_NSE = self.get_joint_NSE_value(next_joint_state)
            joint_state = next_joint_state
            path_joint_states.append(joint_state)
            path_joint_NSE_values.append(joint_NSE)
            joint_NSE_states.append(joint_state)
            joint_NSE_values.append(joint_NSE)
        if report_name != "No Print":
            print(report_name+ ' policy report: ')
            for x in range(len(path_joint_states)):
                print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))
        return joint_NSE_states, path_joint_NSE_values
    
    def get_total_R_and_NSE_from_path(self, Agents:List[WarehouseAgent], path_joint_NSE_values):
        R = 0
        NSE = 0
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        R = [round(agent.R, 2) for agent in Agents]
        NSE = round(float(np.sum(path_joint_NSE_values)), 2)
        print("Total Reward: ", sum(R))
        print("Total NSE: ", NSE)
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
        return R, NSE
    
    def get_blame_DR(self, original_NSE, joint_NSE_state):
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
            counterfactual_constant_state[agent_idx][2] = max(self.Grid.weighting, key=self.Grid.weighting.get)  # most valuable item  
            counterfactual_constant_state[agent_idx] = tuple(counterfactual_constant_state[agent_idx])
            counterfactual_constant_state = tuple(counterfactual_constant_state)
            baseline_performance_by_agent = self.get_joint_NSE_value(counterfactual_constant_state)
            if original_NSE <= baseline_performance_by_agent:
                self.Agents[agent_idx].best_performance_flag = True
            else:
                self.Agents[agent_idx].best_performance_flag = False
            blame_val = round(original_NSE - baseline_performance_by_agent, 2)
            blame[agent_idx] = blame_val + self.epsilon  
        for agent_idx in range(len(self.Agents)):
            NSE_blame[agent_idx] = blame[agent_idx] 
            if np.isnan(NSE_blame[agent_idx]):
                NSE_blame[agent_idx] = 0.0
        return NSE_blame

    def compute_R_Blame_dr_for_all_Agents(self, Agents:List[WarehouseAgent], joint_NSE_states):
        blame_distribution = {}  
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame_DR(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            for agent in Agents:
                blame_array_for_js = -blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_dr[s] = blame_array_for_js[agent.IDX]

    def compute_considerate_reward_for_all_Agents(self, Agents:List[WarehouseAgent], joint_NSE_states):
        blame_distribution = {}  
        alpha_self = 0.5
        alpha_care = 0.5
        for js_nse in joint_NSE_states:
            original_NSE = self.get_joint_NSE_value(js_nse)
            blame_values = self.get_blame(original_NSE, js_nse)
            blame_distribution[js_nse] = np.around(blame_values, 2)
        for js_nse in joint_NSE_states:
            original_NSE = - self.get_joint_NSE_value(js_nse) # just to keep the sign of NSE consistent
            for agent in Agents:
                blame_array_for_js = - blame_distribution[js_nse]
                s = js_nse[agent.IDX]
                agent.R_blame_considerate[s] = alpha_self * blame_array_for_js[agent.IDX] +  alpha_care * (original_NSE - blame_array_for_js[agent.IDX])


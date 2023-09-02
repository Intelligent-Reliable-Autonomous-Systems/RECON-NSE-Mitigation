"""
===============================================================================
This script initialized the grid with goal, obstacles and corresponding actions
also other parameters such as gamma

Factored State Representation ->  < x , y , sample_with_me , coral_flag , samples_at_goal_condition >
    x: denoting x-coordinate of the grid cell
    y: denoting y-coordinate of the grid cell
    sample_with_me: junk unit size being carried by the agent {'X','A','B'}
    coral_flag: boolean denoting presence of coral at the location {True, False}
    sample_at_goal_condition: what the goal deposit looks like to this particular agent
===============================================================================
"""
import copy
import calculation_lib
import read_grid
import numpy as np
from calculation_lib import all_have_reached_goal
from init_agent import Agent
import value_iteration
import simple_colors


class Environment:
    def __init__(self, num_of_agents, goal_deposit, grid_filename, mode, p):

        All_States, rows, columns = read_grid.grid_read_from_file(grid_filename)
        self.all_states = copy.copy(All_States)
        self.file_name = grid_filename
        self.weighting = {'X': 0.0, 'A': 2.0, 'B': 10.0}

        if mode == 'stochastic':
            self.p_success = p
            print(simple_colors.green('mode: STOCHASTIC', ['bold', 'underlined']))
            print(simple_colors.green('p_success = ' + str(self.p_success), ['bold']))
        elif mode == 'deterministic':
            self.p_success = 1.0
            print(simple_colors.green('mode: DETERMINISTIC', ['bold', 'underlined']))
            print(simple_colors.green('p_success = ' + str(self.p_success), ['bold']))
        else:  # defaulting to deterministic
            self.p_success = 1.0
            print(simple_colors.red('UNKNOWN mode: Defaulting to DETERMINISTIC', ['bold', 'underlined']))
            print(simple_colors.red('p_success = ' + str(self.p_success), ['bold']))
            mode = 'deterministic'

        self.mode = mode
        s_goals = np.argwhere(All_States == 'G')
        s_goal = s_goals[0]
        self.s_goal = (s_goal[0], s_goal[1], 'X', False, goal_deposit)

        self.num_of_agents = num_of_agents
        self.S, self.coral_flag = initialize_grid_params(All_States, rows, columns, goal_deposit, self.weighting)
        self.rows = rows
        self.columns = columns
        self.goal_states = s_goals  # this is a list of list [[gx_1,gy_1],[gx_2,gy_2]...]
        self.All_States = All_States
        self.goal_modes = []
        i = 0
        for i in range(int(goal_deposit[0]) + 1):
            self.goal_modes.append((i, 0))
        for j in range(1, int(goal_deposit[1] + 1)):
            self.goal_modes.append((i, j))

    def init_agents_with_initial_policy(self):
        Agents = []
        for label in range(self.num_of_agents):
            Agents.append(Agent((0, 0), self, str(label + 1)))

        # value iteration for all agents
        for agent in Agents:
            agent.V, agent.Pi = value_iteration.value_iteration(agent, self.S)

        # print("[init_env.py] Initial policy for both agents has been computed!!!")
        for agent in Agents:
            agent.s = copy.deepcopy(agent.s0)
            agent.follow_policy()

        return Agents

    def give_joint_NSE_value(self, joint_state):
        # joint_NSE_val = basic_joint_NSE(joint_state)
        # joint_NSE_val = gaussian_joint_NSE(self, joint_state)
        joint_NSE_val = log_joint_NSE(self, joint_state)
        return joint_NSE_val

    def add_goal_reward(self, agent):
        if agent.s == self.s_goal:
            agent.R_blame[agent.s] = 100
            agent.R_blame_dr[agent.s] = 100
        return agent

    def max_log_joint_NSE(self, Agents):
        # this function returns worst NSE in the log NSE formulation
        # state s: < x , y , sample_with_the_agent , coral_flag(True or False), tuple_of_samples_at_goal >
        NSE_worst = 10 * np.log(len(Agents) / 20.0 + 1)
        NSE_worst *= 25  # rescaling it to get good values
        NSE_worst = round(NSE_worst, 2)
        return NSE_worst

    def take_step(self, Grid, Agents):
        NSE_val = 0
        for agent in Agents:
            # print("\n==== Before movement Agent "+agent.label+"'s state: ", agent.s)
            Pi = copy.copy(agent.Pi)
            if agent.s == agent.s_goal:
                # print("Agent "+agent.label+" is at GOAL!")
                continue

            # state of an agent: <x,y,trash_sample_size,coral_flag,samples_at_goal>
            agent.R += agent.Reward(agent.s, Pi[agent.s])
            agent.s = calculation_lib.do_action(agent, agent.s, Pi[agent.s])
            # print("\n==== After movement Agent "+agent.label+"'s state: ", agent.s)
            agent.path = agent.path + " -> " + str(agent.s)
        joint_state = get_joint_state(Agents)
        joint_NSE_val = Grid.give_joint_NSE_value(joint_state)

        return Agents, joint_NSE_val

    def get_blame_reward_by_following_policy(self, Agents):
        NSE_blame_dist = []
        for agent in Agents:
            agent.s = copy.copy(agent.s0)
            RR = 0
            Pi = copy.copy(agent.Pi)
            while agent.s != agent.s_goal:
                RR += agent.R_blame[agent.s]
                agent.s = calculation_lib.do_action(agent, agent.s, Pi[agent.s])
            NSE_blame_dist.append(round(-RR, 2))

        if len(NSE_blame_dist) == 0:
            for _ in range(len(Agents)):
                NSE_blame_dist.append(0.0)

        return NSE_blame_dist


def reset_Agents(Agents):
    for agent in Agents:
        agent.agent_reset()
    return Agents


def compare_all_plans_from_all_methods(Agents):
    for agent in Agents:
        methods = ['Normal initial plan:', 'Rblame mitigation:', 'Generalized mit wo cf data:',
                   'Generalized mit with cf data:']
        print(simple_colors.red("Agent " + agent.label + " plans:", ['bold']))
        while '' in agent.plan_from_all_methods:
            agent.plan_from_all_methods.remove('')
        for i in [0, 1, 2, 3]:
            plan = agent.plan_from_all_methods[i]
            print(simple_colors.yellow(methods[i], ['underlined']))
            print(plan[4:])
        print("=============================")


def show_joint_states_and_NSE_values(Grid, Agents, report_name="No Print"):
    for agent in Agents:
        agent.s = copy.deepcopy(agent.s0)
    path_joint_states = [get_joint_state(Agents)]  # Store the starting joint states
    path_joint_NSE_values = [Grid.give_joint_NSE_value(get_joint_state(Agents))]  # Store the corresponding joint NSE
    joint_NSE_states = []
    joint_NSE_values = []

    while all_have_reached_goal(Agents) is False:
        Agents, joint_NSE = Grid.take_step(Grid, Agents)
        joint_state = get_joint_state(Agents)
        path_joint_states.append(joint_state)
        path_joint_NSE_values.append(joint_NSE)
        joint_NSE_states.append(joint_state)
        joint_NSE_values.append(joint_NSE)

    if report_name != "No Print":
        print(report_name)
        for x in range(len(path_joint_states)):
            print(str(path_joint_states[x]) + ': ' + str(path_joint_NSE_values[x]))

    return joint_NSE_states, path_joint_NSE_values


def get_total_R_and_NSE_from_path(Agents, path_joint_NSE_values):
    R = 0  # Just for storage purposes
    NSE = 0  # Just for storage purposes
    if all_have_reached_goal(Agents):
        print("\nAll Agents have reached the GOAL!!!")
        R = [round(agent.R, 2) for agent in Agents]
        NSE = round(float(np.sum(path_joint_NSE_values)), 2)
        print("Total Reward: ", sum(R))
        print("Total NSE: ", NSE)

    return R, NSE


def initialize_grid_params(All_States, rows, columns, goal_deposit, weighting):
    # Initializing all states
    # s = < x, y, sample_with_me, coral_flag, list_of_samples_at_goal>
    S = []
    goal_modes = []
    i = 0
    for i in range(int(goal_deposit[0]) + 1):
        goal_modes.append((i, 0))
    for j in range(1, int(goal_deposit[1]) + 1):
        goal_modes.append((i, j))
    for i in range(rows):
        for j in range(columns):
            for sample_onboard in weighting.keys():
                for goal_configurations in goal_modes:
                    S.append((i, j, sample_onboard, All_States[i][j] == 'C', goal_configurations))

    # recording coral on the floor
    coral = np.zeros((rows, columns), dtype=bool)
    for i in range(rows):
        for j in range(columns):
            if All_States[i][j] == 'C':
                coral[i][j] = True
            else:
                coral[i][j] = False

    # Now go to value_iteration and change Q function
    # and see how and what to pass when calling it and how to pass probability

    return S, coral


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, np.where(x == v)[0][0]


def check_if_in(arr, Arr):
    return any(np.array_equal(x, arr) for x in Arr)


def get_joint_state(Agents):
    Joint_State = []
    for agent in Agents:
        Joint_State.append(agent.s)
    return tuple(Joint_State)


def log_joint_NSE(Grid, joint_state):
    joint_NSE_val = 0
    X = {}
    for sample_type in Grid.weighting.keys():
        X[sample_type] = 0
    Joint_State = list(copy.deepcopy(joint_state))

    # state s: < x , y , sample_with_agent , coral_flag(True or False), samples_at_goal_condition >
    for s in Joint_State:
        if s[3] is True:
            X[s[2]] += 1
    for sample_type in X.keys():
        joint_NSE_val += Grid.weighting[sample_type] * np.log(X[sample_type] / 20.0 + 1)
    joint_NSE_val *= 25  # rescaling it to get good values
    joint_NSE_val = round(joint_NSE_val, 2)

    return joint_NSE_val

# Grid = Environment({'A': 2, 'B': 3}, 2)
# print(All_States)
# for s in Grid.S:
#     print(s)
# print()
# print("All goal modes: ", Grid.goal_modes)

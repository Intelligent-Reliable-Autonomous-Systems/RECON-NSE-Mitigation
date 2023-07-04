import copy

import calculation_lib
import init_env
from display_lib import display_just_grid


def value_iteration(agent, S):
    """
    :param agent: object of the agent
    :param S: set of all states

    :return V: value V for all states
    """
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
    s_goal = agent.s_goal
    Q = {}
    iterations = 0
    for s in S:
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        iterations += 1
        oldV = V.copy()
        for s in S:
            # print(" -> ", s)
            for a in agent.A[s]:
                # if s == s_goal:
                #     print(" --------------------> ", s)
                #     V[s] = agent.Reward(s, a)
                #     PI[s] = a
                #     continue
                QQ = 0
                # print(" -----> ", s)
                T = calculation_lib.get_transition_prob(agent, s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = QQ + agent.Reward(s, a)
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            print("Value Iteration for Agent " + agent.label + " is done after " + str(iterations) + " Iterations!!")
            break
    return V, PI


def blame_value_iteration(agent, S, R_blame):
    """
    :param agent: object of the agent
    :param S: set of all states
    :param R_blame: Blame reward function

    :return V: value V for all states
    """
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
    s_goal = agent.s_goal
    Q = {}
    iterations = 0
    for s in S:
        # print(s)
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        iterations += 1
        oldV = V.copy()
        for s in S:
            for a in agent.A[s]:
                # if s == s_goal:
                #     V[s] = R_blame[s]
                #     PI[s] = a
                #     continue
                QQ = 0
                T = calculation_lib.get_transition_prob(agent, s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = QQ + R_blame[s]
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            print("Value Iteration for Agent " + agent.label + " is done after " + str(iterations) + " Iterations!!")
            break
    return V, PI


def action_set_value_iteration(agent, S):
    """
    :param agent: object of the agent
    :param S: set of states
    :return policy_space: All optimal actions for all states
    """
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
    s_goal = agent.s_goal
    Q = {}
    iterations = 0
    for s in S:
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        iterations += 1
        oldV = V.copy()
        for s in S:
            for a in agent.A[s]:
                # if s == s_goal:
                #     V[s] = agent.Reward(s, a)
                #     PI[s] = a
                #     continue
                # print("============= A[" + str(s) + "]: " + str(agent.A[s]))
                QQ = 0
                T = calculation_lib.get_transition_prob(agent, s, a)
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = QQ + agent.Reward(s, a)
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            print("Value Iteration for Agent " + agent.label + " is done after " + str(iterations) + " Iterations!!")
            break
    for s in S:
        action_set = [k for k, v in Q[s].items() if round(v, 2) == round(V[s], 2)]
        # print(str(s) + "--------------------> " + str(Q[s]) + " ==== optimal action set: " + str(action_set))
        agent.A[s] = action_set
    return agent


def LVI(Agents, Agents_to_be_corrected, mode):
    """
    :param Agents: All agents
    :param Agents_to_be_corrected: Indices of agents that have been selected to be corrected
    :param mode: 'R_blame' or 'R_blame_gen'
    :return:
    """
    for agent in Agents_to_be_corrected:
        agent = action_set_value_iteration(agent, agent.Grid.S)
        if mode == 'R_blame':
            agent.V, agent.Pi = blame_value_iteration(agent, agent.Grid.S, agent.R_blame)
        elif mode == 'R_blame_gen_wo_cf':
            agent.V, agent.Pi = blame_value_iteration(agent, agent.Grid.S, agent.R_blame_gen_wo_cf)
        elif mode == 'R_blame_gen_with_cf':
            agent.V, agent.Pi = blame_value_iteration(agent, agent.Grid.S, agent.R_blame_gen_with_cf)

    for agent in Agents:
        agent.s = copy.deepcopy(agent.s0)
        agent.follow_policy()

    display_just_grid(Agents[0].Grid.All_States)
    for agent in Agents:
        print("Corrected Plan for Agent " + agent.label + ":")
        print(agent.plan[4:])  # starting for 4 to avoid the initial arrow display ' -> '
        print("________________________________________________\n")

    for agent in Agents:
        agent.agent_reset()

import calculation_lib
import init_env
from init_env import check_if_in
import numpy as np


def value_iteration(agent, S):
    """
    :param agent: object of the agent
    :param S: set of all states

    :return V: value V for all states
    """
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
    s_goal = init_env.s_goal
    Q = {}
    for s in S:
        # print(s)
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        oldV = V.copy()
        for s in S:
            for a in agent.A[s]:
                if s == s_goal:
                    V[s] = agent.Reward(s, a)
                    PI[s] = a
                    continue
                QQ = 0
                T = calculation_lib.get_transition_prob(agent, s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = QQ + agent.Reward(s, a)
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001:
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
    s_goal = init_env.s_goal
    Q = {}
    for s in S:
        # print(s)
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        oldV = V.copy()
        for s in S:
            for a in agent.A[s]:
                if s == s_goal:
                    V[s] = R_blame[s]
                    PI[s] = a
                    continue
                QQ = 0
                T = calculation_lib.get_transition_prob(agent, s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = QQ + R_blame[s]
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001:
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
    s_goal = init_env.s_goal
    Q = {}
    for s in S:
        Q[s] = {}
    for s in S:
        for a in agent.A[s]:
            Q[s][a] = 0.0
    while True:
        oldV = V.copy()
        for s in S:
            for a in agent.A[s]:
                if s == s_goal:
                    V[s] = agent.Reward(s, a)
                    PI[s] = a
                    continue
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

        if max(Residual[s] for s in S) < 0.001:
            break
    for s in S:
        action_set = [k for k, v in Q[s].items() if round(v, 2) == round(V[s], 2)]
        # print(str(s) + "--------------------> " + str(Q[s]) + " ==== optimal action set: " + str(action_set))
        agent.A[s] = action_set
    return agent

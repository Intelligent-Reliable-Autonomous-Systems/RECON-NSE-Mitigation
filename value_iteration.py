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
    s_goal = init_env.s_goals[0]
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
                if (s == (s_goal[0], s_goal[1], 'L', False, ['S']) or s == (
                        s_goal[0], s_goal[1], 'S', False, ['L'])) and a == 'drop':
                    V[s] = agent.Reward(s, a)
                    PI[s] = a
                    continue
                # print("============= A[" + str(s) + "]: " + str(agent.A[s]))
                QQ = 0
                T = calculation_lib.get_transition_prob(s, a)  # returns {s:__, s': __}
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


def action_set_value_iteration(agent, S, R, gamma):
    """
    :param agent: object of the agent
    :param S: set of states
    :param A: set of actions
    :param R: reward function
    :param gamma: Discount factor
    :return policy_space: All optimal actions for all states
    """
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
    s_goals = init_env.s_goals
    QQ = {}
    for s in S:
        QQ[s] = {}
    for s in S:
        for a in agent.A[s]:
            QQ[s][a] = 0.0
    while True:
        oldV = V.copy()
        for s in S:
            if check_if_in(np.array([s[0], s[1], s[2]]), s_goals):
                V[s] = R[s]
                PI[s] = 'G'
                continue
            for a in agent.A[s]:
                # print("============= A[" + str(s) + "]: " + str(agent.A[s]))
                Q = 0
                T = calculation_lib.get_transition_prob(s, a)
                for ss in calculation_lib.get_neighbours(s, a):
                    Q = Q + T[ss] * (gamma * V[ss])
                QQ[s][a] = Q + R[s]
            V[s] = max(QQ[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            act = max(QQ[s], key=QQ[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001:
            break
    for s in S:
        action_set = [k for k, v in QQ[s].items() if round(v, 2) == round(V[s], 2)]
        # print(str(s) + "--------------------> " + str(QQ[s]) + " ==== optimal action set: " + str(action_set))
        agent.A[s] = action_set
    return agent

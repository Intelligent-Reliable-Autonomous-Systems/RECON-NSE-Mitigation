import copy

def value_iteration(agent):
    """
    :param agent: object of the agent
    :param S: set of all states

    :return V: value V for all states
    """
    S = agent.S
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
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
            if s == agent.s_goal:
                V[s] = agent.Reward(s, 'Noop')
                PI[s] = 'Noop'
                Residual[s] = abs(V[s] - oldV[s])
                continue
            for a in agent.A[s]:
                QQ = 0
                T = agent.get_transition_prob(s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = agent.Reward(s, a) + QQ
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            for aa in agent.A[s]:
                Q[s][aa] = round(Q[s][aa])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act
        
        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            # print("Value Iteration for Agent " + agent.label + " is done after " + str(iterations) + " Iterations!!")
            break
    return V, PI

def be_considerate_value_iteration(agent):
    """
    :param agent: object of the agent
    :param S: set of all states

    :return V: value V for all states
    """
    S = copy.deepcopy(agent.Grid.S)
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
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
            if s == agent.s_goal:
                V[s] = agent.Reward(s, 'Noop')
                PI[s] = 'Noop'
                Residual[s] = abs(V[s] - oldV[s])
                continue
            for a in agent.A[s]:
                QQ = 0
                T = agent.get_transition_prob(s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = agent.R_considerate(s, a) + QQ
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            for aa in agent.A[s]:
                Q[s][aa] = round(Q[s][aa])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act

        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            break
    return V, PI


def blame_value_iteration(agent, R_blame):
    """
    :param agent: object of the agent
    :param S: set of all states
    :param R_blame: Blame reward function

    :return V: value V for all states
    """
    S = agent.S
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
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
            if s == agent.s_goal:
                V[s] = agent.Reward(s, 'Noop')
                PI[s] = 'Noop'
                Residual[s] = abs(V[s] - oldV[s])
                continue
            for a in agent.A[s]:
                QQ = 0
                T = agent.get_transition_prob(s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = R_blame[s] + QQ
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            for aa in agent.A[s]:
                Q[s][aa] = round(Q[s][aa])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act
            
        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            break
    return V, PI


def action_set_value_iteration(agent):
    """
    :param agent: object of the agent
    :param S: set of states
    :return policy_space: All optimal actions for all states
    """
    S = agent.S
    V = {s: 0 for s in S}
    Residual = {s: 0 for s in S}
    PI = {s: ' ' for s in S}
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
            if s == agent.s_goal:
                V[s] = agent.Reward(s, 'Noop')
                PI[s] = 'Noop'
                Residual[s] = abs(V[s] - oldV[s])
                continue
            for a in agent.A[s]:
                QQ = 0
                T = agent.get_transition_prob(s, a)  # returns {s:__, s': __}
                for ss in list(T.keys()):
                    QQ = QQ + T[ss] * (agent.gamma * V[ss])
                Q[s][a] = agent.Reward(s, a) + QQ
            V[s] = max(Q[s].values())  # V = max(Q_values)
            Residual[s] = abs(V[s] - oldV[s])
            for aa in agent.A[s]:
                Q[s][aa] = round(Q[s][aa])
            act = max(Q[s], key=Q[s].get)  # Storing the Action corresponding to max Q_value
            PI[s] = act
            
        if max(Residual[s] for s in S) < 0.001 or iterations >= 1000:
            break

    # After computing Q values for all actions in each state, we select all optimal actions
    for s in S:
        action_set = [k for k, v in Q[s].items() if round(v) == round(V[s])]
        agent.A[s] = action_set

    return agent


def LVI(agent, mode):
    if mode == 'considerate':
        V, Pi = be_considerate_value_iteration(agent)
    else:
        agent = action_set_value_iteration(agent)
        if mode == 'recon':
            V, Pi = blame_value_iteration(agent, agent.R_blame)
        elif mode == 'DR':
            V, Pi = blame_value_iteration(agent, agent.R_blame_dr)
        elif mode == 'gen_recon_wo_cf':
            V, Pi = blame_value_iteration(agent, agent.R_blame_gen_wo_cf)
        elif mode == 'gen_recon_w_cf':
            V, Pi = blame_value_iteration(agent, agent.R_blame_gen_with_cf)
        else:
            print("Invalid mode in value iteration!!")
    return V, Pi

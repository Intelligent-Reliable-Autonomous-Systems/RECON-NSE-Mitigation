import value_iteration as vi

def NaivePolicy(agent):
    _, Pi = vi.value_iteration(agent)
    agent.follow_policy(Pi)
    agent.Pi_naive = Pi
    agent.reset()
    return Pi

def ReconPolicy(agent):
    _, Pi = vi.LVI(agent, 'recon')
    agent.follow_policy(Pi)
    agent.Pi_recon = Pi
    agent.reset()
    return Pi

def GenRecon_wo_cf_Policy(agent):
    _, Pi = vi.LVI(agent, 'gen_recon_wo_cf')
    agent.follow_policy(Pi)
    agent.Pi_gen_recon_wo_cf = Pi
    agent.reset()
    return Pi

def GenRecon_w_cf_Policy(agent):
    _, Pi = vi.LVI(agent, 'gen_recon_w_cf')
    agent.follow_policy(Pi)
    agent.Pi_gen_recon_with_cf = Pi
    agent.reset()
    return Pi

def DR_Policy(agent):
    _, Pi = vi.LVI(agent, 'DR')
    agent.follow_policy(Pi)
    agent.Pi_dr = Pi
    agent.reset()
    return Pi

def ConsideratePolicy(agent):
    _, Pi = vi.LVI(agent, 'considerate')
    agent.follow_policy(Pi)
    agent.Pi_considerate = Pi
    agent.reset()
    return Pi




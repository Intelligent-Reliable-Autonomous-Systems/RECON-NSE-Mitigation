from display import *

def get_all_visualizations(save_figures=True):
    domains = ['salp', 'overcooked', 'warehouse']

    # Figure 1: Varying #agents undergoing policy update
    for domain in domains:
        NSE_naive = np.loadtxt('sim_results/' + domain + '/Vary_NSE_Naive.txt', dtype=float) 
        NSE_dr = np.loadtxt('sim_results/' + domain + '/Vary_NSE_DR.txt', dtype=float)
        NSE_recon = np.loadtxt('sim_results/' + domain + '/Vary_NSE_RECON.txt', dtype=float)
        NSE_gen_recon_with_cf = np.loadtxt('sim_results/' + domain + '/Vary_NSE_GEN_RECON_with_cf.txt', dtype=float)
        NSE_considerate = np.loadtxt('sim_results/' + domain + '/Vary_NSE_CONSIDERATE.txt', dtype=float)
        agents_corrected = np.loadtxt('sim_results/' + domain + '/Vary_agent_percentage_corrected.txt', dtype=int)
        plot_NSE_LinePlot_with_corrected_agents(NSE_naive, NSE_dr, NSE_recon, NSE_gen_recon_with_cf, NSE_considerate, agents_corrected, domain, save_fig=save_figures)

    # Figure 2: NSE Penalty for different techniques with increasing number of agents
    for domain in domains:
        NSE_naive_tracker = np.loadtxt('sim_results/' + domain + '/NSE_naive_tracker.txt', dtype=float)
        NSE_considerate_tracker = np.loadtxt('sim_results/' + domain + '/NSE_considerate_tracker.txt', dtype=float)
        NSE_recon_tracker = np.loadtxt('sim_results/' + domain + '/NSE_recon_tracker.txt', dtype=float)
        NSE_gen_recon_wo_cf_tracker = np.loadtxt('sim_results/' + domain + '/NSE_gen_recon_wo_cf_tracker.txt', dtype=float)
        NSE_gen_recon_with_cf_tracker = np.loadtxt('sim_results/' + domain + '/NSE_gen_recon_with_cf_tracker.txt', dtype=float)
        NSE_dr_tracker = np.loadtxt('sim_results/' + domain + '/NSE_dr_tracker.txt', dtype=float)
        num_of_agents_tracker = np.loadtxt('sim_results/' + domain + '/num_of_agents_tracker.txt', dtype=int)
        plot_effect_of_generalization(NSE_naive_tracker, NSE_recon_tracker, NSE_gen_recon_wo_cf_tracker,
                                        NSE_gen_recon_with_cf_tracker, NSE_dr_tracker, NSE_considerate_tracker, 
                                        num_of_agents_tracker, domain, save_fig=save_figures)



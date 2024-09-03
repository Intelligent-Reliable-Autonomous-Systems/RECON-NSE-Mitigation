import varying_corrected_agent as sim1
import effect_of_generalization as sim2
import get_plots as sim3

def main():
    #domain = 0: salp, 1: overcooked, 2: warehouse
    for domain in [1]:#[0, 1, 2]:
        if domain == 0:
            print('Domain: Salp')
            domain_name = 'salp'
            from salp_mdp import SalpAgent as Agent
            from salp_mdp import SalpEnvironment as Environment
            from metareasoner import SalpMetareasoner as MR
        elif domain == 1:
            print('Domain: Overcooked')
            domain_name = 'overcooked'
            from overcooked_mdp import OvercookedAgent as Agent
            from overcooked_mdp import OvercookedEnvironment as Environment
            from metareasoner import OvercookedMetareasoner as MR
        elif domain == 2:
            print('Domain: Warehouse')
            domain_name = 'warehouse'
            from warehouse_mdp import WarehouseAgent as Agent
            from warehouse_mdp import WarehouseEnvironment as Environment
            from metareasoner import WarehouseMetareasoner as MR
        else:
            print('Domain not recognized')
            exit()

        # sim1.run_varying_corrected_agents_sim(domain_name, Agent, Environment, MR)
        sim2.run_generalization_simulation(domain_name, Agent, Environment, MR)
        sim3.get_all_visualizations(save_figures=True)

if __name__ == '__main__':
    main()
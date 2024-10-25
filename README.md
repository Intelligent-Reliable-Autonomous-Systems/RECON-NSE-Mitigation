# NSE Mitigation in Multi-Agent Systems using Blame Assignment
We model the problem of mitigating NSEs in a cooperative multi-agent system as a Lexicographic Decentralized Markov Decision Process with two objectives. The agents must optimize the completion of their assigned tasks while mitigating  NSEs. We assume independence of transitions and rewards with respect to the agents' tasks but the joint NSE penalty creates a form of dependence in this setting. To improve scalability, the joint NSE penalty is decomposed into individual penalties for each robot using credit assignment, which facilitates decentralized policy computation. We empirically demonstrate using mobile robots and in simulation the effectiveness and scalability of our approach in mitigating NSEs by updating the policies of a subset of agents in the system.

## Installation
After cloning the repo, setup a conda environment with python3, activate it, and install requirement.txt
```bash
$ pip install -r requirements.txt
```
## Run Simulation
```bash
$ python main.py
```
## Results will be saved in the `sim_results/<domain>` folder
All simulation results including figures for all domains will be generated an saved in respective locations (in the `sim_results/<domain>`folder) after running the main simulation

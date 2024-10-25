# Mitigating Side Effects in Multi-Agent Systems Using Blame Assignment
When independently trained or designed robots are deployed in a shared environment, their combined actions can lead to unintended negative side effects (NSEs). To ensure safe and efficient operation, robots must optimize task performance while minimizing the penalties associated with NSEs, balancing individual objectives with collective impact. We model the problem of mitigating NSEs in a cooperative multi-agent system as a bi-objective lexicographic decentralized Markov decision process. We assume independence of transitions and rewards with respect to the robots' tasks, but the joint NSE penalty creates a form of dependence in this setting. To improve scalability, the joint NSE penalty is decomposed into individual penalties for each robot using credit assignment, which facilitates decentralized policy computation. We empirically demonstrate, using mobile robots and in simulation, the effectiveness and scalability of our approach in mitigating NSEs.

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

# Reinforcement Learning - Practical 2
## How to run
Install requirements:
```bash
cd source code
pip install -r requirements.txt
```
Run part 1 (Prisoner's Dilemma):
```bash
python train_iql.py # Saves plots in a .pdf
python train_cql.py # Saves plots in a .pdf
```
Run part 2 (LBF):
```bash
python lbf_experiment.py # Saves plots in a .pdf, videos in a .gif, and checkpoints in a .pkl
```
## Videos for part 2
### Non-cooperative
IQL Non-cooperative:
![IQL Non-cooperative](source%20code/iql_non_coop.gif)  
CQL Non-cooperative:
![CQL Non-cooperative](source%20code/cql_non_coop.gif)  
### Cooperative
IQL Cooperative:
![IQL Cooperative](source%20code/iql_coop.gif)  
CQL Cooperative:
![CQL Cooperative](source%20code/cql_coop.gif)  
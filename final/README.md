# README of our optimizer

## Requirements and Installation 

Since our tool uses SMAC as a code base, it has the same requirements. Therefore just follow the installation instructions of SMAC (included in the readme inside the tool folder) in order to setup the tool. 
No further requirements are needed. 

## Running our tool

Our tool can be started either the same way as SMAC is started by calling the SMAC script 

```
python tool/scripts/smac --scenario_file <scenario.txt> [--parallel_options <CL+EACH, CL+LIST, EF+EACH, EF+LIST, UCB+EACH, UCB+List]> [--num_cores <1,2,4,8>] --SEED <int>
```

We added two parameters, one for controlling the amount of parallel runs which can be chosen from 1,2,4 and 8 and the second one for controlling which scenario to choose. CL stands short for constant liar, EF for expectations across fantasies and UCB for itself. The two racing methods are EACH for evaluating after each instance and LIST for evaluating only after a list of instances. Any combination of the 3 and 2 can be chosen as an option, combined with a +. 

The other way is from the inside of aclib2. We included all the necessary files that need to be added to aclib2 in order to add our tool as a new configurator. We added six different configurators, one for each scenario. 

## Results 


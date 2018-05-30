from smac.configspace import ConfigurationSpace, Configuration
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.utils.validate import Validator
from smac.utils.io.traj_logging import TrajLogger
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import RunKey
import numpy as np
from matplotlib import pyplot as plt


smac_runs = []

for i in range(0, 1):
    # Build a scenario
    scenario = Scenario("scenario.txt")

    # ###### Configuration part ############

    # Create SMAC object
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42))

    # optimize with SMAC
    best_conf = smac.optimize()
    print("*********************Incumbent Found:")
    print(best_conf)

    # ###### Validation part ##############
    traj_logger = TrajLogger(None, Stats(scenario))
    trajectory = traj_logger.read_traj_aclib_format("smac-output/run_1/traj_aclib2.json", scenario.cs)
    # print(trajectory)
    validator = Validator(scenario, trajectory, rng=np.random.RandomState(42))

    # evaluate on test instances and calculate cpu time
    runhis_dev = validator.validate(config_mode="def", instance_mode="test")
    runhis_inc = validator.validate(config_mode="inc", instance_mode="test")

    # print("*********************Runhistory Dev of Validation Part:")
    # print(runhis_dev.data)
    # print("*********************Runhistory Inc of Validation Part:")
    # print(runhis_inc.data)

    # copied from the smac documentation, is not included in Runhistory anymore
    def get_instance_costs_for_config(runhis: RunHistory, config: Configuration):
        """
        Returns the average cost per instance (across seeds)
            for a configuration
            Parameters
            ----------
            config : Configuration from ConfigSpace
                Parameter configuration

            Returns
            -------
            cost_per_inst: dict<instance name<str>, cost<float>>
        """
        config_id = runhis.config_ids.get(config)
        runs_ = runhis._configid_to_inst_seed.get(config_id, [])
        cost_per_inst = {}
        for inst, seed in runs_:
            cost_per_inst[inst] = cost_per_inst.get(inst, [])
            rkey = RunKey(config_id, inst, seed)
            vkey = runhis.data[rkey]
            cost_per_inst[inst].append(vkey.cost)
        cost_per_inst = dict([(inst, np.mean(costs)) for inst, costs in cost_per_inst.items()])
        return cost_per_inst

    default_conf = runhis_dev.ids_config[1]
    incumbent_conf = runhis_inc.ids_config[1]
    dev_vals = get_instance_costs_for_config(runhis_dev, default_conf)
    inc_vals = get_instance_costs_for_config(runhis_inc, incumbent_conf)

    # ###### Filter runs for plotting #######
    dev_x = []
    inc_x = []
    for key in set(dev_vals.keys()) & set(inc_vals.keys()):
        dev_x.append(dev_vals[key])
        inc_x.append(inc_vals[key])

    # print(dev_vals)
    # print(inc_vals)
    print(dev_x)
    print(inc_x)

    fig, ax = plt.subplots()
    ax.scatter(dev_x, inc_x, marker="x")
    ax.set_xlabel("Default Configuration")
    ax.set_ylabel("Incumbent Configuration")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.savefig("fig-2.png")

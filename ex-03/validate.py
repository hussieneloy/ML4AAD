from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException
from smac.configspace import ConfigurationSpace, Configuration
from smac.utils.validate import Validator
from smac.utils.io.traj_logging import TrajLogger
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import RunKey
import numpy as np
import logging
import sys
from matplotlib import pyplot as plt


class RunGurobi(object):

    def __init__(self):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def run(self, scen):
        optimizer = SMAC(scenario=scen)

        try:
            optimizer.optimize()
        except (TAEAbortException, FirstRunCrashedException) as err:
            self.logger.error(err)

    def validate(self, scenario):
        traj_logger = TrajLogger(None, Stats(scenario))
        trajectory = traj_logger.read_traj_aclib_format("traj_aclib2.json", scenario.cs)
        
        validator = Validator(scenario, trajectory, rng=np.random.RandomState(42))

        # evaluate on test instances and calculate cpu time
        runhis_dev = validator.validate(config_mode="def", instance_mode="test")
        runhis_inc = validator.validate(config_mode="inc", instance_mode="test")

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

        # ### Save to file ###
        f = open("results.txt", "w")
        f.write("Dev Results: \n" + str(dev_x) + "\n")
        f.write("Inc Results: \n" + str(inc_x) + "\n")
        f.close()

        # ### Plotting, needs to be uncommented with import pyplot at top for cluster ###
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

        # ax.set_xscale('log')
        # ax.set_yscale('log')

        fig.savefig("result.png")


if __name__=="__main__":
    folder = sys.argv[1]
    scen = Scenario("scenario.txt")
    run = RunGurobi()
    run.run(scen)
    # print("Done with optimizing. Start validating now.")
    print(sys.argv)
    run.validate(scen)

import time
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt

from smac.configspace import Configuration
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunKey, RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from es_facade import ES


def get_instance_costs_for_config(runhis: RunHistory, config: Configuration):
    """
    return average cost per instance
    :param runhis: SMAC run history
    :param config: parameter configuration
    :return: mapping from instance name to cost
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


def optimize(optimizer, scenario, trajectory=None):
    then = time.time()
    best_conf = optimizer.optimize()
    print(best_conf)
    print('training   time:', time.time() - then)

    traj_logger = TrajLogger(None, Stats(scenario))
    trajectory = trajectory or traj_logger.read_traj_aclib_format("smac-output/run_1/traj_aclib2.json", scenario.cs)
    validator = Validator(scenario, trajectory, rng=np.random.RandomState(42))

    # evaluate on test instances and calculate cpu time
    then = time.time()
    runhis_dev = validator.validate(config_mode="def", instance_mode="test")
    runhis_inc = validator.validate(config_mode="inc", instance_mode="test")
    print('validating time:', time.time() - then)

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

    print('PAR10:', np.mean(inc_x), '/', np.mean(dev_x))
    max_x = 1000.0
    par1er = lambda xx: np.mean([(x / 10 if x == max_x else x) for x in xx])
    print('PAR1 :', par1er(inc_x), '/', par1er(dev_x))
    to_counter = lambda xx: len([x for x in xx if x == max_x])
    print('TOs  :', to_counter(inc_x), '/', to_counter(dev_x))
    print('wins :', len([i for i in range(len(dev_x)) if dev_x[i] > inc_x[i]]), '/', len(dev_x))

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

    fig.savefig("fig-smac.png")


def smackdown_optimize(args):
    scenario = Scenario('scenario.txt')
    optimizer = ES(
        scenario=scenario,
        rng=np.random.RandomState(12),
        run_id=1,
        parallel_options=args.next + '+' + args.parallel,
        cores=args.cores
    )
    # best_conf = optimizer.optimize()
    # print(best_conf)
    optimize(optimizer, scenario)


def smac_optimize():
    scenario = Scenario('scenario.txt')
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42))
    optimize(smac, scenario)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('runner')
    parser.add_argument('-p', '--parallel', help='Parallelism method', choices=['EACH', 'LIST', 'INDP'], default='LIST')
    parser.add_argument('-c', '--cores', help='Number of CPU cores to use', type=int, default=2)
    parser.add_argument('-n', '--next', help='Method of choosing next configs',
                        choices=['CL', 'UBC', 'FA', 'TS'], default='CL')
    smackdown_optimize(parser.parse_args())

import argparse
import json
import numpy as np

from smac.facade.func_facade import fmin_smac

from mllab import find_cut_off


def run_smac(python_path, w_dir, n_iter=5, input_file='../rawAllx1000.json', seeds=[1], task_ids=None, max_tries=10):

    def test_func(cutoff):
        result = find_cut_off.main(python_path=python_path, w_dir=w_dir, iter=n_iter, input_file=input_file,
                                   cutoffs=[cutoff], seeds=seeds, task_ids=task_ids)
        cleaned = [x[1] for x in result if 0.0 < x[1] < 1.0]
        mean = np.mean(cleaned) if cleaned else 0.0
        mean = mean if mean != 1.0 else 0.0
        return 1.0 - mean

    x, cost, smac = fmin_smac(func=test_func,
                              x0=[5],  # default values
                              bounds=[(1, 99)],  # bounds of each x
                              maxfun=max_tries,  # maximal iterations
                              rng=1234  # random seed
                              )

    return x, cost, smac


def run_roar(python_path, w_dir, n_iter=5, input_file='../rawAllx1000.json', seeds=[1], task_ids=None, max_tries=10):

    from smac.configspace import ConfigurationSpace
    from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
    from smac.scenario.scenario import Scenario
    from smac.facade.roar_facade import ROAR

    def test_func(cutoff):
        cutoff = cutoff.get('cutoff')
        print(cutoff)
        result = find_cut_off.main(python_path=python_path, w_dir=w_dir, iter=n_iter, input_file=input_file,
                                   cutoffs=[cutoff], seeds=seeds, task_ids=task_ids)
        cleaned = [x[1] for x in result if 0.0 < x[1] < 1.0]
        mean = np.mean(cleaned) if cleaned else 0.0
        mean = mean if mean != 1.0 else 0.0
        return 1.0 - mean

    cs = ConfigurationSpace()
    cutoff_parameter = UniformIntegerHyperparameter('cutoff', 1, 99, default_value=50)
    cs.add_hyperparameter(cutoff_parameter)
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_tries,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "abort_on_first_run_crash": "false",
                         })

    roar = ROAR(scenario=scenario, tae_runner=test_func, rng=1234)

    x = roar.optimize()

    cost = test_func(x)

    return x, cost, roar


def plot(smac, plot=False):
    runhistory = smac.get_runhistory()

    # extract x value and corresponding y value
    x_smac = []
    y_smac = []
    for entry in runhistory.data:  # iterate over data because it is an OrderedDict
        config_id = entry.config_id  # look up config id
        config = runhistory.ids_config[config_id]  # look up config
        y_ = runhistory.get_cost(config)  # get cost
        x_ = config["x1"]  # there is only one entry in our example
        if 0.0 < y_ < 1.0:
            x_smac.append(x_)
            y_smac.append(1 - y_)
        with open('benchmark-x.json', 'w') as f:
            json.dump(x_smac, f)
        with open('benchmark-y.json', 'w') as f:
            json.dump(y_smac, f)

    if plot:
        x_smac = np.array(x_smac)
        y_smac = np.array(y_smac)
        p = x_smac.argsort()
        import matplotlib.pyplot as plt
        plt.plot(x_smac[p], y_smac[p])
        plt.grid()
        plt.show()


def clean_smac_shit():
    import os
    import shutil
    for f in os.listdir('.'):
        if f.startswith('smac3-output_'):
            shutil.rmtree(f)


def main(args):
    run_func = run_smac if args.method == 'SMAC' else run_roar
    x, y, smac = run_func(python_path=args.python_path, w_dir=args.working_dir, input_file=args.input_file,
                          n_iter=args.iter, seeds=args.seed, task_ids=args.tasks, max_tries=args.smac_iter)
    plot(smac)
    clean_smac_shit()


if __name__ == '__main__':
    calc_parser = argparse.ArgumentParser('benchmark')
    calc_parser.add_argument('-i', '--iter', default=5, type=int, help='number of iterations on each of the datasets')
    calc_parser.add_argument('-I', '--smac-iter', default=10, type=int, help='number of SMAC iterations')
    calc_parser.add_argument('-s', '--seed', default=[1], type=int, nargs='+', help='random seeds')
    calc_parser.add_argument('-t', '--tasks', default=None, type=str, nargs='+', help='task IDs')
    calc_parser.add_argument('-S', '--save', default='cutoff.json', type=str, help='output json file')
    calc_parser.add_argument('-f', '--input-file', default='rawAllx1000.json', help='the input json file')
    calc_parser.add_argument('-p', '--python-path', default='/usr/bin', type=str, help='absolute path of python exec')
    calc_parser.add_argument('-w', '--working-dir', default='.', type=str, help='path of working directory')
    calc_parser.add_argument('-m', '--method', choices=['SMAC', 'ROAR'], default='SMAC')
    main(calc_parser.parse_args())

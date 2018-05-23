import argparse
import numpy as np
import matplotlib.pyplot as plt

from smac.facade.func_facade import fmin_smac

from mllab import find_cut_off


def run_smac(python_path, w_dir, iter=5, input_file='../rawAllx1000.json', seeds=[1], task_ids=None):

    def test_func(cutoff):
        result = find_cut_off.main(python_path=python_path, w_dir=w_dir, iter=iter, input_file=input_file,
                                   cutoffs=[cutoff], seeds=seeds, task_ids=task_ids)
        return np.mean([y for _, y in result.items()])

    # x, cost, smac = fmin_smac(func=test_func,
    #                           x0=[-0],  # default values
    #                           bounds=[(-5, 5)],  # bounds of each x
    #                           maxfun=,  # maximal number of function evaluations
    #                           rng=1234  # random seed
    #                           )
    #
    # return x, cost, smac

    return test_func(5)


def clean_smac_shit():
    import os
    import shutil
    for f in os.listdir('.'):
        if f.startswith('smac3-output_'):
            shutil.rmtree(f)


def main(args):
    run_smac(python_path=args.python_path, w_dir=args.working_dir, input_file=args.input_file,
             iter=args.iter, seeds=args.seed, task_ids=args.tasks)


if __name__ == '__main__':
    calc_parser = argparse.ArgumentParser('benchmark')
    calc_parser.add_argument('-i', '--iter', default=5, type=int, help='number of iterations on each of the datasets')
    calc_parser.add_argument('-s', '--seed', default=[1], type=int, nargs='+', help='random seeds')
    calc_parser.add_argument('-t', '--tasks', default=None, type=str, nargs='+', help='task IDs')
    calc_parser.add_argument('-S', '--save', default='cutoff.json', type=str, help='output json file')
    calc_parser.add_argument('-f', '--input-file', default='rawAllx1000.json', help='the input json file')
    calc_parser.add_argument('-p', '--python-path', default='/usr/bin', type=str, help='absolute path of python exec')
    calc_parser.add_argument('-w', '--working-dir', default='.', type=str, help='path of working directory')
    main(calc_parser.parse_args())

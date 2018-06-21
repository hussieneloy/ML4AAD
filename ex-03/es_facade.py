import typing
import numpy as np
import logging

from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.intensification.intensification import Intensifier
from smac.utils.io.traj_logging import TrajLogger
from smac.configspace import Configuration
from smac.optimizer.objective import average_cost
from smac.utils.constants import MAXINT
from smac.utils.io.output_directory import create_output_directory

from optimizer import ESOptimizer


class ES(object):
    """ Facade to use ES

    """

    def __init__(self,
                 scenario: Scenario,
                 tae_runner: typing.Union[ExecuteTARun, typing.Callable]=None,
                 stats: Stats=None,
                 runhistory: RunHistory=None,
                 intensifier: Intensifier=None,
                 rng: typing.Union[np.random.RandomState, int]=None,
                 run_id: int=1,
                 X: int=10,
                 M: int=10,
                 A: int=3,
                 initialPop: int=20,
                 extension: bool=False):

        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        aggregate_func = average_cost

        self.output_dir = create_output_directory(scenario, run_id)
        scenario.write()

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        # initialize empty runhistory
        if runhistory is None:
            runhistory = RunHistory(aggregate_func=aggregate_func)
        # inject aggr_func if necessary
        if runhistory.aggregate_func is None:
            runhistory.aggregate_func = aggregate_func
        
        # initial random number generator
        num_run, rng = self._get_rng(rng=rng)

        # initial Trajectory Logger
        traj_logger = TrajLogger(output_dir=self.output_dir, stats=self.stats)

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        if tae_runner is None:
            tae_runner = ExecuteTARunOld(ta=scenario.ta,
                                         stats=self.stats,
                                         run_obj=scenario.run_obj,
                                         runhistory=runhistory,
                                         par_factor=scenario.par_factor,
                                         cost_for_crash=scenario.cost_for_crash)
        # Second case, the tae_runner is a function to be optimized
        elif callable(tae_runner):
            tae_runner = ExecuteTAFuncDict(ta=tae_runner,
                                           stats=self.stats,
                                           run_obj=scenario.run_obj,
                                           memory_limit=scenario.memory_limit,
                                           runhistory=runhistory,
                                           par_factor=scenario.par_factor,
                                           cost_for_crash=scenario.cost_for_crash)
        else:
            raise TypeError("Target algorithm not supported. Must be either a call "
                            "string in the scenario file or a callable.")

        # Check that overall objective and tae objective are the same
        if tae_runner.run_obj != scenario.run_obj:
            raise ValueError("Objective for the target algorithm runner and "
                             "the scenario must be the same, but are '%s' and "
                             "'%s'" % (tae_runner.run_obj, scenario.run_obj))

        # inject stats if necessary
        if tae_runner.stats is None:
            tae_runner.stats = self.stats
        # inject runhistory if necessary
        if tae_runner.runhistory is None:
            tae_runner.runhistory = runhistory
        # inject cost_for_crash
        if tae_runner.crash_cost != scenario.cost_for_crash:
            tae_runner.crash_cost = scenario.cost_for_crash

        # initialize intensification
        if intensifier is None:
            intensifier = Intensifier(tae_runner=tae_runner,
                                      stats=self.stats,
                                      traj_logger=traj_logger,
                                      rng=rng,
                                      instances=scenario.train_insts,
                                      cutoff=scenario.cutoff,
                                      deterministic=scenario.deterministic,
                                      run_obj_time=scenario.run_obj == "runtime",
                                      always_race_against=scenario.cs.get_default_configuration() \
                                        if scenario.always_race_default else None,
                                      instance_specifics=scenario.instance_specific,
                                      minR=scenario.minR,
                                      maxR=scenario.maxR)

        # inject deps if necessary
        if intensifier.tae_runner is None:
            intensifier.tae_runner = tae_runner
        if intensifier.stats is None:
            intensifier.stats = self.stats
        if intensifier.traj_logger is None:
            intensifier.traj_logger = traj_logger

        # TODO: look into initial design/configuration, random or default
        # TODO: runhistory2epm for extension (ex2)

        es_args = {
            'scenario': scenario,
            'stats': self.stats,
            # 'initial_design': initial_design,
            'runhistory': runhistory,
            # 'runhistory2epm': runhistory2epm,
            'intensifier': intensifier,
            'aggregate_func': aggregate_func,
            # 'num_run': num_run,
            # 'model': model,
            # 'acq_optimizer': acquisition_function_optimizer,
            # 'acquisition_func': acquisition_function,
            'rng': rng,
            'X': X,
            'M': M,
            'A': A,
            'initialPop': initialPop,
            'extension': extension
        }

        self.solver = ESOptimizer(**es_args)

    def optimize(self):
        self.solver.run()

    def validate(self):
        pass

    def _get_rng(self, rng):
        """Initialize random number generator

        If rng is None, initialize a new generator
        If rng is Int, create RandomState from that
        If rng is RandomState, return it

        Parameters
        ----------
        rng: np.random.RandomState|int|None

        Returns
        -------
        int, np.random.RandomState
        """
        # initialize random number generator
        if rng is None:
            self.logger.debug('no rng given: using default seed of 1')
            num_run = 1
            rng = np.random.RandomState(seed=num_run)
        elif isinstance(rng, int):
            num_run = rng
            rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            num_run = rng.randint(MAXINT)
            rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))
        return num_run, rng

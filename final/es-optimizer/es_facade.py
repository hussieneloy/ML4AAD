import typing
import numpy as np
import logging
import os
import shutil

from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType
from smac.utils.util_funcs import get_types
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.intensification.intensification import Intensifier
from smac.utils.io.traj_logging import TrajLogger
from smac.epm.rfr_imputator import RFRImputator
from smac.configspace import Configuration
from smac.optimizer.objective import average_cost
from smac.epm.base_epm import AbstractEPM
from smac.utils.constants import MAXINT
from smac.optimizer.acquisition import EI, LogEI, AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer
from smac.epm.rf_with_instances import RandomForestWithInstances

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
                 acquisition_function: AbstractAcquisitionFunction=None,
                 acquisition_function_optimizer: AcquisitionFunctionMaximizer=None,
                 model: AbstractEPM=None,
                 runhistory2epm: AbstractRunHistory2EPM=None,
                 rng: typing.Union[np.random.RandomState, int]=None,
                 run_id: int=1,
                 parallel_options: str=None,
                 cores: int=2):

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

        def intensifier_maker(instants):
            return Intensifier(tae_runner=tae_runner,
                               stats=self.stats,
                               traj_logger=traj_logger,
                               rng=rng,
                               instances=instants,
                               cutoff=scenario.cutoff,
                               deterministic=scenario.deterministic,
                               run_obj_time=scenario.run_obj == "runtime",
                               always_race_against=scenario.cs.get_default_configuration() \
                                   if scenario.always_race_default else None,
                               instance_specifics=scenario.instance_specific,
                               minR=scenario.minR,
                               maxR=scenario.maxR)

        if intensifier is None:
            intensifier = intensifier_maker(scenario.train_insts)

        # inject deps if necessary
        if intensifier.tae_runner is None:
            intensifier.tae_runner = tae_runner
        if intensifier.stats is None:
            intensifier.stats = self.stats
        if intensifier.traj_logger is None:
            intensifier.traj_logger = traj_logger
        
        if parallel_options is None:
            parallel_options = "CL+LIST"

        # initial conversion of runhistory into EPM data
        if runhistory2epm is None:

            num_params = len(scenario.cs.get_hyperparameters())
            if scenario.run_obj == "runtime":

                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(scenario.cutoff)
                threshold = np.log10(scenario.cutoff *
                                     scenario.par_factor)

                imputor = RFRImputator(rng=rng,
                                       cutoff=cutoff,
                                       threshold=threshold,
                                       model=model,
                                       change_threshold=0.01,
                                       max_iter=2)

                runhistory2epm = RunHistory2EPM4LogCost(
                    scenario=scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=True,
                    impute_state=[StatusType.CAPPED, ],
                    imputor=imputor)

            elif scenario.run_obj == 'quality':
                runhistory2epm = RunHistory2EPM4Cost(scenario=scenario, num_params=num_params,
                                                     success_states=[
                                                         StatusType.SUCCESS, 
                                                         StatusType.CRASHED],
                                                     impute_censored_data=False, impute_state=None)

            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)

        # inject scenario if necessary:
        if runhistory2epm.scenario is None:
            runhistory2epm.scenario = scenario

        # initial EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        if model is None:
            model = RandomForestWithInstances(types=types, bounds=bounds,
                                              instance_features=scenario.feature_array,
                                              seed=rng.randint(MAXINT),
                                              pca_components=scenario.PCA_DIM)
        # initial acquisition function
        if acquisition_function is None:
            if scenario.run_obj == "runtime":
                acquisition_function = LogEI(model=model)
            else:
                acquisition_function = EI(model=model)
        # inject model if necessary
        if acquisition_function.model is None:
            acquisition_function.model = model

        # initialize optimizer on acquisition function
        if acquisition_function_optimizer is None:
            acquisition_function_optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function, scenario.cs, np.random.RandomState(seed=rng.randint(MAXINT))
            )
        elif not isinstance(
                acquisition_function_optimizer,
                AcquisitionFunctionMaximizer,
            ):
            raise ValueError(
                "Argument 'acquisition_function_optimizer' must be of type"
                "'AcquisitionFunctionMaximizer', but is '%s'" %
                type(acquisition_function_optimizer)
            )

        es_args = {
            'scenario': scenario,
            'stats': self.stats,
            # 'initial_design': initial_design,
            'runhistory': runhistory,
            'runhistory2epm': runhistory2epm,
            'intensifier': intensifier,
            'aggregate_func': aggregate_func,
            # 'num_run': num_run,
            'model': model,
            'acq_optimizer': acquisition_function_optimizer,
            'acquisition_func': acquisition_function,
            'rng': rng,
            'parallel_options': parallel_options,
            'intensifier_maker': intensifier_maker,
            'cores': cores,
            'traj_logger': traj_logger,
        }

        self.solver = ESOptimizer(**es_args)

    def optimize(self):
        incumbent = None
        try:
            incumbent = self.solver.run()
        finally:
            self.solver.stats.save()
            self.solver.stats.print_stats()
            self._logger.info("Final Incumbent: %s" % (self.solver.incumbent))
            self.runhistory = self.solver.runhistory
            self.trajectory = self.solver.intensifier.traj_logger.trajectory

            if self.output_dir is not None:
                self.solver.runhistory.save_json(
                    fn=os.path.join(self.output_dir, "runhistory.json")
                )
        return incumbent

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

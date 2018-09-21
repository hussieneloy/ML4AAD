import os
import itertools
import logging
import numpy as np
import random
import time
import datetime
import typing
import math
import multiprocessing
import ConfigSpace
import threading
import copy
import multiprocessing

from multiprocessing import Process, Queue, Pool, Manager
from rx import Observable
from rx.concurrency import ThreadPoolScheduler
from smac.configspace import Configuration, get_one_exchange_neighbourhood
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier
from smac.optimizer import pSMAC
from smac.optimizer.acquisition import AbstractAcquisitionFunction, LogEI, EI
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer, RandomSearch, LocalSearch
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import FirstRunCrashedException, BudgetExhaustedException
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator
from smac.optimizer.objective import _cost
from smac.utils.util_funcs import get_types
from sklearn import linear_model, preprocessing
from collections import Counter
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run import ExecuteTARun
from smac.runhistory.runhistory import RunHistory
from smac.utils.constants import MAXINT
from smac.tae.execute_ta_run import StatusType
from smac.configspace.util import convert_configurations_to_array
from ConfigSpace.util import get_random_neighbor
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, IntegerHyperparameter
from ConfigSpace.hyperparameters import FloatHyperparameter, UniformIntegerHyperparameter, Constant



__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


class SMBO(object):

    """Interface that contains the main Bayesian optimization loop

    Attributes
    ----------
    logger
    incumbent
    scenario
    config_space
    stats
    initial_design
    runhistory
    rh2EPM
    intensifier
    aggregate_func
    num_run
    model
    acq_optimizer
    acquisition_func
    rng
    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 num_parallel: int,
                 parallel_scenario: str,
                 traj_logger: TrajLogger,
                 restore_incumbent: Configuration=None):
        """
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory
            data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration
            (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a
             configuration
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances)
        acq_optimizer: AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill
            criterion for acq_optimizer)
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng
        self.num_parallel = num_parallel
        self.traj_logger = traj_logger
        self.parallel_scenario = parallel_scenario

        self._random_search = RandomSearch(
            acquisition_func, self.config_space, rng
        )
        self.manager = Manager()
        self.winner_dict = self.manager.dict()
        self.racing_results = self.manager.list()
        self.rh_entries = self.manager.list()
        self.stat_runs = multiprocessing.Value('i', 0)
        self.stat_runtime = multiprocessing.Value('d', 0.0)

    def start(self):
        """Starts the Bayesian Optimization loop.
        Detects whether we the optimization is restored from previous state.
        """
        self.stats.start_timing()
        # Initialization, depends on input
        if self.stats.ta_runs == 0 and self.incumbent is None:
            try:
                self.incumbent = self.initial_design.run()
            except FirstRunCrashedException as err:
                if self.scenario.abort_on_first_run_crash:
                    raise
        elif self.stats.ta_runs > 0 and self.incumbent is None:
            raise ValueError("According to stats there have been runs performed, "
                             "but the optimizer cannot detect an incumbent. Did "
                             "you set the incumbent (e.g. after restoring state)?")
        elif self.stats.ta_runs == 0 and self.incumbent is not None:
            raise ValueError("An incumbent is specified, but there are no runs "
                             "recorded in the Stats-object. If you're restoring "
                             "a state, please provide the Stats-object.")
        else:
            # Restoring state!
            self.logger.info("State Restored! Starting optimization with "
                             "incumbent %s", self.incumbent)
            self.logger.info("State restored with following budget:")
            self.stats.print_stats()

    def run(self):
        """Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        """
        self.start()

        self.incumbent = self.scenario.cs.get_default_configuration()
        challengers = []
        num_insts = 10

        # Main BO loop
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_dirs=self.scenario.input_psmac_dirs,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")

            # get all found configurations sorted according to acq
            #challengers = self.choose_next(X, Y).challengers
            
            cores = self.num_parallel
            if "EACH" in self.parallel_scenario:
                cores = 10

            if "CL" in self.parallel_scenario:
                challengers = self.cl(challengers, cores)
            elif "EF" in self.parallel_scenario:
                challengers = self.expect_fan(challengers, cores)
            else: # ucb
                challengers = self.ucb(challengers, cores)

            if type(challengers) is not list:
                challengers = [challengers]

            # time_spent = time.time() - start_time
            # time_left = self._get_timebound_for_intensification(time_spent)

            self.logger.debug("Intensify")
            self.logger.info("###############START RACING################")

            if "EACH" in self.parallel_scenario:
                challengers.append(self.incumbent)
                self.incumbent, inc_perf = self.race_irace_like(challengers)
            else: # LIST
                incumbent_runs = self.runhistory.get_runs_for_config(self.incumbent)
                if len(incumbent_runs) == 1:
                    challengers[cores - 1] = self.incumbent
                self.incumbent, inc_perf = self.list_racing(challengers, num_insts)

            self.logger.info("###############STOP RACING################")
            self.logger.info("CURRENT INCUMBENT:")
            # self.logger.info(self.incumbent)

            num_insts += 1

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir_for_this_run)

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    """def __writer(self, queue, challengers, instance):
        # print("INSIDE WRITER FUNCTION")
        for config in challengers:
            queue.put((config, instance))
            # print("Adding tuple (%s,%s) to queue" % (config, instance))
        for i in range(self.num_parallel):
            queue.put("DONE")"""

    """def __reader(self, queue, i):
        while True:
            msg = queue.get()
            if msg == "DONE":
                print("Done, stopping process %s" % (i))
                break
            else:
                print("WORKING ON PROCESS %s" % (i))
                config, cost = self.run_config_on_instance(msg[0], msg[1])
                # print("RESULT: (%s, %s)" % (config, cost))
                self.racing_results.append((config, cost))
                # print("RACING RESULTS:")
                # print(self.racing_results)"""

    """def race_irace_like(self, challengers):
        # if more challengers than instances, take as many challengers as instances exist
        instances = random.sample(self.scenario.train_insts, len(challengers))
        # print("INSTANCES")
        # print(instances)
        # print("CHALLENGERS:")
        # print(challengers)
        for inst in instances:
            # print("CHALLENGERS:")
            # print(challengers)
            # print("############### WORKING ON INSTANCE %s ############" % (inst))
            # create parallel stuff

            self.racing_results = self.manager.list()
            queue = Queue()
            proc = []

            if self.stats.is_budget_exhausted():
                self.logger.info("Time budget exhaustet, cancel Racing, return incumbent")
                return self.incumbent, 1

            for i in range(0, self.num_parallel):
                print("STARTED PROCESS %s" % (i))
                p = Process(target=self.__reader, args=((queue), (i)))
                p.deamon = True
                p.start()
                proc.append(p)

            self.__writer(queue, challengers, inst)
            # print("AFTER PROCESS ADDED")
            for pr in proc:
                pr.join()

            # print("AFTER PROCESSES JOINT")
            # print("RESULT LIST:")
            # print(results)
            self.update_rh_and_stat()
            loser = max(self.racing_results, key=lambda t: t[1])
            # print("LOSER:")
            # print(loser)
            if len(challengers) > 1:
                challengers.remove(loser[0])
            # print("NEW CHALLENGERS:")
            # print(challengers)
            if len(challengers) == 1:
                break
            
        if challengers[0] is not self.incumbent:
            self.stats.inc_changed += 1
        print("TA RUNS: %s" % (self.stats.ta_runs))
        return challengers[0], 1"""

    """def initializer(self):
        global tae_runner
        tae_runner = ExecuteTARun(self.scenario.ta, self.stats, self.runhistory, self.scenario.run_obj)"""

    """def race_irace_like_pool(self, challengers):
        # if more challengers than instances, take as many challengers as instances exist
        instances = random.sample(self.scenario.train_insts, len(challengers))
        for inst in instances:
            # print("############### WORKING ON INSTANCE %s ############" % (inst))
            # create parallel stuff
            self.racing_results = self.manager.list()
            queue = Queue()
            proc = []
            pool = Pool(processes=self.num_parallel, initializer=self.initializer)

            results = pool.map(self.run_config_on_instance, [(config, inst) for config in challengers])
            # results = [p.get() for p in output]

            print("RESULT LIST:")
            print(results)
            self.update_rh_and_stat()
            loser = max(results, key=lambda t: t[1])
            # print("LOSER:")
            # print(loser)
            if len(challengers) > 1:
                challengers.remove(loser[0])
            print("NEW CHALLENGERS:")
            print(challengers)
            if len(challengers) == 1:
                break
        if challengers[0] is not self.incumbent:
            self.stats.inc_changed += 1
        print("TA RUNS: %s" % (self.stats.ta_runs))
        return challengers[0], 1"""

    def race_irace_like(self, challengers):
        # if more challengers than instances, take as many challengers as instances exist
        if len(challengers) > len(self.scenario.train_insts):
            challengers = challengers[:len(self.scenario.train_insts) - 1]
            challengers.append(self.incumbent)
        instances = self.scenario.train_insts[:len(challengers)]

        for inst in instances:

            self.racing_results = self.manager.list()
            queue = Queue()

            for pair in ((i, inst) for i in challengers):
                queue.put(pair)
            for i in range(self.num_parallel):
                queue.put("DONE")

            processes = [Process(target=self.race, args=((queue), )) for x in range(self.num_parallel)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            self.update_rh_and_stat()
            self.logger.info(self.racing_results)
            loser = max(self.racing_results, key=lambda t: t[1])
            self.logger.info(loser)
            if len(challengers) > 1:
                challengers.remove(loser[0])
            print("NEW CHALLENGERS:")
            print(challengers)
            print("#######################")
            if len(challengers) == 1:
                break

        winner = challengers[0]
        winner_cost = self.runhistory.get_cost(winner)
        # print("WINNERS COST: %s" % (winner_cost))
        incumbent_cost = self.runhistory.get_cost(self.incumbent)
        # print("INCUMBENT COST: %s" % (incumbent_cost))
        winner_runs = self.runhistory.get_runs_for_config(winner)
        incumbent_runs = self.runhistory.get_runs_for_config(self.incumbent)
        # print("WINNER RUNS: %s" % (winner_runs))
        # print("INCUMBENT RUNS: %s" % (incumbent_runs))
        winner_cost /= float(len(winner_runs))
        incumbent_cost /= float(len(incumbent_runs))
        
        if winner_cost < incumbent_cost:
            self.stats.inc_changed += 1
            self.traj_logger.add_entry(winner_cost,
                                       self.runhistory.config_ids[winner],
                                       winner)
            self.logger.info("Changed Incumbent")
            return winner, winner_cost
        else:
            self.logger.info("Incumbent won the racing.")
            return self.incumbent, incumbent_cost

    def race(self, queue):
        while True:
            pair = queue.get()
            # print("GOT PAIR FROM QUEUE AT %s" % (datetime.datetime.now()))
            # print(pair)
            if pair == "DONE":
                # print("QUEUE EMPTY AT %s" % (datetime.datetime.now()))
                break
            else:
                (conf, cost) = self.run_config_on_instance(pair[0], pair[1])
                self.racing_results.append((conf, cost))
                # print("RACING RESULTS:")
                # print(self.racing_results)

    def update_rh_and_stat(self):
        self.stats.ta_runs = self.stat_runs.value
        self.stats.ta_time_used = self.stat_runtime.value
        # print("RUNTIME: %s" % (self.stats.ta_time_used))
        for entry in self.rh_entries:
            config = entry[0]
            cost = entry[1]
            runtime = entry[2]
            status = entry[3]
            instance = entry[4]
            seed = entry[5]
            additional_info = entry[6]
            self.runhistory.add(config=config,
                                cost=cost, time=runtime, status=status,
                                instance_id=instance, seed=seed,
                                additional_info=additional_info)
        self.rh_entries = self.manager.list()

    def run_config_on_instance(self, challenger, instance):
        # tae_runner = ExecuteTARun(self.scenario.ta, self.stats, self.runhistory, self.scenario.run_obj)
        # print("TARGET ALGO CALL")
        status, cost, runtime, additional_info = self.intensifier.tae_runner.run(
                challenger,
                instance,
                self.scenario.cutoff,
                12345,
                self.scenario.instance_specific.get(instance, "0"))

        self.rh_entries.append((challenger, cost, runtime, status, instance, 12345, additional_info))
        self.stat_runs.value += 1
        self.stat_runtime.value += float(runtime)
        return (challenger, cost) 
    
    def list_racing(self, challengers, num_insts):
        train_insts = self.scenario.train_insts
        instances = train_insts[:num_insts]
        # The randomized way
        """
        insts_sz = len(train_insts)
        challengers_num = len(challengers)
        num_insts = min(num_insts, insts_sz)
        random_insts_perm = np.random.permutation(insts_sz)
        random_insts_perm = random_insts_perm[:num_insts]
                for idx in random_insts_perm:
            instances.append(train_insts[idx])
        """
        proc = []

        self.event = multiprocessing.Event()

        for challenger in challengers:
            p = Process(target=self.run_config_on_instances, args=(challenger,instances))
            p.daemon = True
            p.start()
            proc.append(p)

        while True:
            if self.event.is_set():
                for p in proc:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.1)

        self.update_rh_and_stat()
        self.event.clear()

        winner = self.winner_dict['winner']

        winner_cost = self.runhistory.get_cost(winner)
        incumbent_cost = self.runhistory.get_cost(self.incumbent)
        winner_runs = self.runhistory.get_runs_for_config(winner)
        incumbent_runs = self.runhistory.get_runs_for_config(self.incumbent)
        
        winner_cost /=  float(len(winner_runs))
        incumbent_cost /=  float(len(incumbent_runs))
        
        if winner_cost < incumbent_cost:
            self.stats.inc_changed += 1
            self.traj_logger.add_entry(winner_cost,
                                       self.runhistory.config_ids[winner],
                                       winner)
            self.logger.info("Changed Incumbent")
            return winner, winner_cost
        else:
            return self.incumbent, incumbent_cost

    def run_config_on_instances(self, challenger, instances):
        total_costs = 0
        # tae_runner = ExecuteTARun(self.scenario.ta, self.stats, self.runhistory,
        #                          self.scenario.run_obj)
        for inst in instances:
            status, cost, runtime, additional_info = self.intensifier.tae_runner.run(
                    challenger,
                    inst,
                    self.scenario.cutoff,
                    12345,
                    self.scenario.instance_specific.get(inst, "0"))
            total_costs += cost

            self.rh_entries.append((challenger, cost, runtime, status, inst, 12345, additional_info))
            self.stat_runs.value += 1
            self.stat_runtime.value += float(runtime)    
        self.winner_dict['winner'] = challenger

        print("Found Winner with cost %s" % (total_costs / len(instances)))
#        print(self.winner_dict['winner'])
        self.event.set()
        return challenger, total_costs / len(instances)

    def choose_next(self, X: np.ndarray, Y: np.ndarray,
                    incumbent_value: float=None):
        """Choose next candidate solution with Bayesian optimization. The 
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        incumbent_value: float
            Cost value of incumbent configuration
            (required for acquisition function);
            if not given, it will be inferred from runhistory;
            if not given and runhistory is empty,
            it will raise a ValueError

        Returns
        -------
        Iterable
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )

        self.model.train(X, Y)

        if incumbent_value is None:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent_value = self.runhistory.get_cost(self.incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        challengers = self.acq_optimizer.maximize(
            self.runhistory, self.stats, 5000
        )
        return challengers

    def cl(self, configs, ret_num):
        if len(self.runhistory.data) == 1:
            return self.scenario.cs.sample_configuration(size=ret_num)

        # Get all instances available
        instances = self.scenario.train_insts
        # A copy from the runhistory in order not to ruin the original with lies
        copied_data = dict(self.runhistory.data)
        tmp_runhistory = RunHistory(aggregate_func=self.aggregate_func)
        tmp_runhistory.data = copied_data
        tmp_runhistory.ids_config = self.runhistory.ids_config
        tmp_runhistory.config_ids = self.runhistory.config_ids
        tmp_runhistory.cost_per_config = self.runhistory.cost_per_config
        tmp_runhistory.runs_per_config = self.runhistory.runs_per_config
      
        for config in configs:
            # Get runs of the current configuration
            conf_runs = self.runhistory.get_runs_for_config(config)
            conf_runs = [x[0] for x in conf_runs]
            # Get the available runs for the current configuration
            # available_insts = instances - conf_runs
            available_insts = [x for x in instances if x not in conf_runs]
            if len(available_insts) == 0:
                available_insts = instances
            # Choose random instance
            random_idx = np.random.randint(0, len(available_insts))
            random_inst = list(available_insts)[random_idx]
            # Getting the cost of the configuration runs
            conf_costs = _cost(config, self.runhistory) 
            # Get the CL-min, the minimum variation of Constant-liar
            lie = np.amin(conf_costs) if len(conf_costs) > 0 else 1.0

            # Add hallucination to the fake runhistory
            tmp_runhistory.add(config=config, cost=lie, 
            time=0.01, status=StatusType.SUCCESS, instance_id=random_inst)
        
        # Training the EPM with the temporary runhistory
        X, Y = self.rh2EPM.transform(tmp_runhistory)
        self.model.train(X, Y)
        incumbent_value = self.runhistory.get_cost(self.incumbent)
        # Updating the acqusition function and invoking the maximizer
        self.acquisition_func.update(model=self.model, eta=incumbent_value)
        new_challengers = self.acq_optimizer.maximize(tmp_runhistory, self.stats, 50)

        # Picking the best candidates
        answer = new_challengers.challengers[:ret_num]
        return answer

    def validate(self, config_mode='inc', instance_mode='train+test',
                 repetitions=1, use_epm=False, n_jobs=-1, backend='threading'):
        """Create validator-object and run validation, using
        scenario-information, runhistory from smbo and tae_runner from intensify

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms (in
            deterministic will be fixed to 1)
        use_epm: bool
            whether to use an EPM instead of evaluating all runs with the TAE
        n_jobs: int
            number of parallel processes used by joblib

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs
        """
        traj_fn = os.path.join(self.scenario.output_dir_for_this_run, "traj_aclib2.json")
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=self.scenario.cs)
        new_rh_path = os.path.join(self.scenario.output_dir_for_this_run, "validated_runhistory.json")

        validator = Validator(self.scenario, trajectory, self.rng)
        if use_epm:
            new_rh = validator.validate_epm(config_mode=config_mode,
                                            instance_mode=instance_mode,
                                            repetitions=repetitions,
                                            runhistory=self.runhistory,
                                            output=new_rh_path)
        else:
            new_rh = validator.validate(config_mode, instance_mode, repetitions,
                                        n_jobs, backend, self.runhistory,
                                        self.intensifier.tae_runner,
                                        output=new_rh_path)
        return new_rh

    def _get_timebound_for_intensification(self, time_spent):
        """Calculate time left for intensify from the time spent on
        choosing challengers using the fraction of time intended for
        intensification (which is specified in
        scenario.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.scenario.intensification_percentage
        if frac_intensify <= 0 or frac_intensify >= 1:
            raise ValueError("The value for intensification_percentage-"
                             "option must lie in (0,1), instead: %.2f" %
                             (frac_intensify))
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug("Total time: %.4f, time spent on choosing next "
                          "configurations: %.4f (%.2f), time left for "
                          "intensification: %.4f (%.2f)" %
                          (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left

    def expect_fan(self, configs, ret_num):
        if len(self.runhistory.data) == 1:
            return self.scenario.cs.sample_configuration(size=ret_num)

        # Filling The aggregate acquisition values with zeros
        sum_acq = np.zeros(10)
        for config in configs:
            # Getting the cost of the configuration runs
            conf_costs = _cost(config, self.runhistory)
            if len(conf_costs) == 0:
                continue
 
            x, y = [], []
            for (xi, yi) in enumerate(conf_costs):
                x.append([xi])
                y.append(yi)
            x = np.array(x)
            y = np.array(y)
 
            # Predicting the next 10 values using linear regression
            regr = linear_model.LinearRegression()
            regr.fit(x, y)
            x_test = [[i] for i in range(len(x), 10 + len(x))]
            x_test = np.ndarray.flatten(np.array(x_test))
            x_test = [float(i) for i in x_test]
            x_test = np.array([[x] for x in x_test])
            posterior = regr.predict(x_test)
 
            types, bounds = np.array([0]), np.array([[0.0, 1.0]])
            posterior = np.array([[y] for y in posterior])
            # Fitting the EPM with the posterior values
            epm_model = RandomForestWithInstances(types=types, bounds=bounds,
                                              instance_features=None,
                                              seed=12345,pca_components=12345,
                                              ratio_features=1,
                                              num_trees=1000,
                                              min_samples_split=1,
                                              min_samples_leaf=1,
                                              max_depth=100000,
                                              do_bootstrapping=False,
                                              n_points_per_tree=-1,
                                              eps_purity=0)
 
            # Training Acquistion function with the new model
            post_acq = EI(model=epm_model)
            epm_model.train(x_test, posterior) 
            incumbent_value = np.amin(posterior)
            post_acq.update(model=epm_model, eta=incumbent_value)
            # Adding the acquisition values to the aggregate function
            acq_values = post_acq._compute(X=x_test)[:,0]
            sum_acq = [a + b for a, b in zip(sum_acq, acq_values)]
 
        # Constructing the Final EPM which trains the aggregate function
        types, bounds = get_types(self.scenario.cs, self.scenario.feature_array)
        all_configs = self.runhistory.get_all_configs()
        conf_array = convert_configurations_to_array(all_configs)
        upd_sz = len(conf_array[0])
        types = types[:upd_sz]
        sum_model = RandomForestWithInstances(types=types, bounds=bounds,
                                              instance_features=None,
                                              seed=12345,pca_components=12345,
                                              ratio_features=1,
                                              num_trees=1000,
                                              min_samples_split=1,
                                              min_samples_leaf=1,
                                              max_depth=100000,
                                              do_bootstrapping=False,
                                              n_points_per_tree=-1,
                                              eps_purity=0)
        x_axis = np.arange(10)
        x_axis = np.array([[float(x)] for x in x_axis])
        sum_acq = np.array([[y] for y in sum_acq])
 
        X, Y = self.rh2EPM.transform(self.runhistory)
 
        new_x = []
        x_idx = 0
        size_x = len(X)
        while len(new_x) < 10:
            new_x.append(X[x_idx % size_x][:upd_sz])
            x_idx += 1
        new_x = np.array(new_x)
        sum_model.train(new_x, sum_acq)
        incumbent_value = self.runhistory.get_cost(self.incumbent)

        # Updating the acquistion function with aggregate epm model
        self.acquisition_func.update(model=sum_model, eta=incumbent_value)

        new_challengers = self.acq_optimizer.maximize(self.runhistory, self.stats, 50)

        # Picking the best performing configuration
        answer = new_challengers.challengers[:ret_num]

        return answer

    def ucb(self, configs, ret_num):
        if len(self.runhistory.data) == 1:
            return self.scenario.cs.sample_configuration(size=ret_num)

        new_configs = []

        config_ucbs = []
        for config in configs:
            conf_costs = _cost(config, self.runhistory)

            if len(conf_costs) == 0:
                config_ucbs.append((config, MAXINT))
                continue

            cur_beta = np.random.random_sample()
            ucb_value = self.ucb_func(conf_costs, cur_beta)
            config_ucbs.append((config, ucb_value))

        config_ucbs.sort(key=lambda x: x[1])
        best_conf = config_ucbs[0][0]
        best_neighbours = get_one_exchange_neighbourhood(best_conf,
                            seed=self.rng.randint(MAXINT))
        neighbour_list = [best_conf]

        for neighbour in best_neighbours:
            if len(neighbour_list) == ret_num:
                break

            neighbour_list.append(neighbour)

        return neighbour_list

    def ucb_func(self, values, beta):
        miu = np.mean(values)
        sigma = np.std(values)
        return miu + beta * sigma

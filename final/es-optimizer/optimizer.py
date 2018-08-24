import numpy as np
import threading
import time
import math
import random
import logging
import copy
from sklearn import linear_model, preprocessing


import multiprocessing
from rx import Observable
from rx.concurrency import ThreadPoolScheduler
import ConfigSpace
import math
from collections import Counter
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.configspace import Configuration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.utils.util_funcs import get_types
from smac.utils.constants import MAXINT
from smac.tae.execute_ta_run import StatusType
from smac.scenario.scenario import Scenario
from smac.configspace import convert_configurations_to_array
from smac.stats.stats import Stats
from smac.optimizer.acquisition import AbstractAcquisitionFunction, LogEI, EI
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer, RandomSearch, LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import _cost
from smac.utils.io.traj_logging import TrajLogger
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from ConfigSpace.util import get_random_neighbor
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, IntegerHyperparameter
from ConfigSpace.hyperparameters import FloatHyperparameter, UniformIntegerHyperparameter, Constant


class ESOptimizer(object):
    """Interface that contains the parallel optimizer

    Attributes:

    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 parallel_options: str,
                 cores: int,
                 intensifier_maker=lambda x: ()):

        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.incumbent = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.stats = stats
        self.cores = cores
        self.rh2EPM = runhistory2epm
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.parallel_options = parallel_options
        self.intensifier_maker = intensifier_maker

    def start(self):
        self.stats.start_timing()
        self._logger.info("Start time: %s" % (self.stats._start_time))

    def run(self):
        self.start()


        # Give a chance for the default confiugration
        # default config is set as starting incumbent
        self._logger.info("Default Configuration: %s" % (self.incumbent))
        # Add 5 random configuration at first
        challengers = self.scenario.cs.sample_configuration(size=5)
        # Add a run for the incumbent configuration to be used in optimization.
        self.intensifier._add_inc_run(incumbent=self.incumbent, run_history=self.runhistory)
        while True:

            start_time = time.time()
            
            # Culling the number of challengers
            if len(challengers) == 10:
                challengers = self.scenario.cs.sample_configuration(size=5)

            # Find configurations to evaluate next
            self._logger.info("Searching for next configuration")
            challenger = self.choose_next(challengers)
            challengers.append(challenger)

            time_spent = time.time() - start_time
            self._logger.info("Time spent for choosing configuration: %s" % (time_spent))

            # Race configurations
            time_left = self._get_timebound_for_intensification(time_spent)
            self._logger.info("Time left for intensification: %s" % (time_left))
            self._logger.info("Intensifying...")
            
            self.incumbent, inc_perf = self.race_configs(challengers, self.incumbent, time_left)
            if self.incumbent in challengers:
                challengers.remove(self.incumbent)
                challengers.append(self.scenario.cs.sample_configuration())
            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self, configs):
        # Constant Liar
        if "CL+" in self.parallel_options:
            return self.cl(configs)
        # UBC
        elif "UBC+" in self.parallel_options:
            return self.ucb(configs)
        # Expectations across fantasies
        elif "FA+" in self.parallel_options:
            return self.expect_fan(configs)
        # Thompson Sampling
        elif "TS+" in self.parallel_options:
            pass
        else:
            ValueError("Wrong Combination Type")

        # dummy solution: just sample random configs
        challengers = self.scenario.cs.sample_configuration(size=5)

        return challengers

    def race_configs(self, set_of_conf, incumbent, time_left):
        """Races the challengers agains each other to determine incumbent
        """
        start_time = time.time()
        time_bound = max(self.intensifier._min_time, time_left, 10)
        list_of_champions = []  # containing pairs of the form (incumbent, performance)
        # optimal_thread_count = multiprocessing.cpu_count()
        # pool_scheduler = ThreadPoolScheduler(optimal_thread_count - 1)
        pool_scheduler = ThreadPoolScheduler(self.cores)

        def local_race(intensifier):
            def local_race_with_intensifier(c):
                if time.time() - start_time > time_bound:
                    return c, np.infty
                return intensifier.intensify(
                    challengers=[c],
                    incumbent=incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=time_bound
                )
            return local_race_with_intensifier

        try:
            # Run all in parallel for list of instances
            if "+LIST" in self.parallel_options:
                logging.info('Race different configurations in parallel')
                for conf in set_of_conf:
                    for instance in self.scenario.train_insts:
                        Observable.of(conf).map(local_race(self.intensifier_maker([instance]))) \
                            .subscribe_on(pool_scheduler) \
                            .subscribe(lambda x: list_of_champions.append(x))
                while len(list_of_champions) == 0:
                    time.sleep(time_bound)
                    print('number of champions:', len(list_of_champions))
                return min(list_of_champions, key=lambda x: x[1])
            # Run all in parallel for each instance
            elif "+EACH" in self.parallel_options:
                logging.info('Race different configurations in parallel')
                for conf in set_of_conf:
                    Observable.of(conf).map(local_race(self.intensifier)) \
                        .subscribe_on(pool_scheduler) \
                        .subscribe(lambda x: list_of_champions.append(x))
                while len(list_of_champions) == 0:
                    time.sleep(time_bound)
                    print('number of champions:', len(list_of_champions))
                return min(list_of_champions, key=lambda x: x[1])
            # Independent race against incumbent
            elif "+INDP" in self.parallel_options:
                pass
            else:
                ValueError("Wrong Combination Type")

            # dummy solution
            # best, inc_perf = self.intensifier.intensify(
            #     challengers=set_of_conf,
            #     incumbent=incumbent,
            #     run_history=self.runhistory,
            #     aggregate_func=self.aggregate_func,
            #     time_bound=max(self.intensifier._min_time, time_left)
            # )
            # return best, inc_perf
        except TypeError as e:
            raise e

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
        return time_left

    def cl(self, configs):
        # Get all instances available
        instances = self.intensifier.instances
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
            available_insts = instances - set(conf_runs)
            if len(available_insts) == 0:
                available_insts = instances
            # Choose random instance
            random_idx = np.random.randint(0, len(available_insts))
            random_inst = list(available_insts)[random_idx]
            # Getting the cost of the configuration runs
            conf_costs = _cost(config, self.runhistory) 
            # Get the CL-min, the minimum variation of Constant-liar
            lie = np.amin(conf_costs) if (len(conf_costs) > 0) else 0

            # Add hallucination to the fake runhistory
            tmp_runhistory.add(config=config, cost=lie, 
            time=0.0, status=StatusType.SUCCESS, instance_id=random_inst)
        
        
        # Training the EPM with the temporary runhistory
        X, Y = self.rh2EPM.transform(tmp_runhistory)
        self.model.train(X, Y)
        incumbent_value = self.runhistory.get_cost(self.incumbent)
        # Updating the acqusition function and invoking the maximizer
        self.acquisition_func.update(model=self.model, eta=incumbent_value)
        new_challengers = self.acq_optimizer.maximize(tmp_runhistory, self.stats, 50)

        # Picking the best candidate
        answer = new_challengers.challengers[0]
        return answer

    def expect_fan(self, configs):
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
        types, bounds = np.array([0]), np.array([[0.0, 1.0]])
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
        sum_model.train(x_axis, sum_acq)
        incumbent_value = self.runhistory.get_cost(self.incumbent)
        
        # Updating the acquistion function with aggregate epm model
        self.acquisition_func.update(model=sum_model, eta=incumbent_value)

        new_challengers = self.acq_optimizer.maximize(self.runhistory, self.stats, 50)

        # Picking the best performing configuration
        answer = new_challengers.challengers[0]
        
        return answer

    def ucb(self, configs):
        new_configs = []
        # This loop should be replaced with parallel workers
        for config in configs:
            new_config, new_cost = self.single_ucb(config)
            new_configs.append((new_config, new_cost))
        best = min(new_configs, key= lambda t:t[1])
        return best[0]


    def single_ucb(self, config):
        conf_costs = _cost(config, self.runhistory)
        config_runs = len(conf_costs)
        if config_runs == 0:
            random_config = self.scenario.cs.sample_configuration()
            return random_config, 0.0
        types, bounds = np.array([0]), np.array([[0.0, 1.0]])
        local_model = RandomForestWithInstances(types=types, bounds=bounds,
                                              instance_features=None,
                                              seed=12345,pca_components=12345,
                                              ratio_features=1,
                                              num_trees=1000,
                                              min_samples_split=1,
                                              min_samples_leaf=1,
                                              max_depth=100000,
                                              do_bootstrapping=True,
                                              n_points_per_tree=-1,
                                              eps_purity=0)
        local_acq = EI(model=local_model)
        local_optimizer = LocalSearch(local_acq, self.scenario.cs, 
            np.random.RandomState(seed=self.rng.randint(MAXINT)))
        tae_runner = ExecuteTARunOld(ta=self.scenario.ta,
                                         stats=self.stats,
                                         run_obj=self.scenario.run_obj,
                                         runhistory=self.runhistory,
                                         par_factor=self.scenario.par_factor,
                                         cost_for_crash=self.scenario.cost_for_crash)
        x_axis = np.arange(config_runs)
        x_axis = np.array([[float(x)] for x in x_axis])
        costs = np.array([[y] for y in conf_costs])
        local_model.train(x_axis, costs)
        min_so_far = np.amin(conf_costs)
        local_acq.update(model=local_model, eta=min_so_far)
        possible_neighbors = local_optimizer.maximize(self.runhistory, self.stats, 10)
        cand_costs = np.zeros(10)
        conf_insts = []
        for i in range(10):
            conf_insts.append([])
        beta_t = 0.2
        instances = self.intensifier.instances
        for iteration in range(5):
            rand_inst = random.sample(instances, 1)
            for cand in range(10):
                cur_cand = possible_neighbors[cand]
                seed = 0
                if self.scenario.deterministic == 0:
                    seed = self.rng.randint(low=0, high=MAXINT, size=1)[0]
                try:
                    status, cost, dur, res = tae_runner.start(
                    config=cur_cand,
                    instance=rand_inst,
                    seed=seed,
                    cutoff=self.scenario.cutoff,
                    instance_specific=self.scenario.instance_specific.get(
                        inst, "0"))
                except:
                    self.stats.print_stats(debug_out=True)
                    break
                conf_insts[cand].append(cost)
                ucb_sofar = self.ucb_func(conf_insts[cand], beta_t)
                cand_costs[cand] += ucb_sofar
            beta_t += 0.2
        best_idx = np.argmin(cand_costs)
        return possible_neighbors[best_idx], cand_costs[best_idx]

        

    def ucb_func(self, values, beta):
        miu = np.mean(values)
        sigma = np.std(values)
        return miu + beta * sigma

    """def generate_random_configuration(self):

        #The function generates random configurations by randomizing each
        #hyparameters in the configuration until a valid configuration is
        #returned.

        hyper_params = self.scenario.cs.get_hyperparameters()
        candidate_conf = None
        found = False
        while not found:
            values = dict()
            for param in hyper_params:
                key, val = self.random_parameter(param)
                values[key] = val
            print(values)
            try:
                candidate_conf = Configuration(configuration_space=self.scenario.cs,
                                               values=values,
                                               allow_inactive_with_values=True)
                found = True
            except ValueError as e:
                continue
        return candidate_conf"""

    """def random_parameter(self, param):
        # The function accepts a hyperparameter and returns random
        # value accordingly.
        if isinstance(param, Constant):
            return param.name, param.value
        elif isinstance(param, FloatHyperparameter):
            a = param.lower
            b = param.upper
            val = (b - a) * np.random.random_sample() + a
            return param.name, val
        elif isinstance(param, IntegerHyperparameter):
            a = param.lower
            b = param.upper
            val = np.random.random_integers(a, b)
            return param.name, val
        elif isinstance(param, CategoricalHyperparameter):
            return param.name, random.choice(list(param.choices))
        else:
            idx = np.random.random_integers(0, param._num_choices)
            choices_list = list(sequence)
            val = choices_list[idx]
            return param.name, val"""

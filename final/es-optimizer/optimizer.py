import numpy as np
import threading
import time
import math
import random
import logging

import multiprocessing
from rx import Observable
from rx.concurrency import ThreadPoolScheduler
import ConfigSpace
import math
from collections import Counter
from smac.configspace import Configuration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from ConfigSpace.util import get_random_neighbor
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, IntegerHyperparameter
from ConfigSpace.hyperparameters import FloatHyperparameter, UniformIntegerHyperparameter, Constant


class ESOptimizer(object):
    """Interface that contains the main Evolutionary optimizer

    Attributes:

    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 rng: np.random.RandomState,
                 parallel_options: str,
                 intensifier_maker=lambda x: ()):

        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.incumbent = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.intensifier = intensifier
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

        while True:

            start_time = time.time()

            # Find configurations to evaluate next
            self._logger.info("Searching for next configuration")
            challengers = self.choose_next()

            time_spent = time.time() - start_time
            self._logger.info("Time spent for choosing configuration: %s" % (time_spent))

            # Race configurations
            time_left = self._get_timebound_for_intensification(time_spent)
            self._logger.info("Time left for intensification: %s" % (time_left))
            self._logger.info("Intensifying...")
            self.incumbent, inc_perf = self.race_configs(challengers, self.incumbent, time_left)

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self):
        # Constant Liar
        if "CL+" in self.parallel_options:
            pass
        # UBC
        elif "UBC+" in self.parallel_options:
            pass
        # Expectation acroo fantasies
        elif "FA+" in self.parallel_options:
            pass
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
        optimal_thread_count = multiprocessing.cpu_count()
        pool_scheduler = ThreadPoolScheduler(optimal_thread_count - 1)

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

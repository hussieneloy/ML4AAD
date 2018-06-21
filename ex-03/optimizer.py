import numpy as np
import threading
import time
import math
import random
import logging

import ConfigSpace
import math
from smac.configspace import Configuration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from ConfigSpace.util import get_random_neighbor
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter
from ConfigSpace.hyperparameters import FloatHyperparameter, UniformIntegerHyperparameter, Constant
from PopMember import PopMember


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
                 X: int=10,
                 M: int=10,
                 A: int=3,
                 initialPop: int=20,
                 extension: bool=False):

        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.incumbent = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.nc_pop = []
        self.c_pop = []
        self.attractive = []
        self.unattractive = []
        self.X = X 
        self.M = M
        self.A = A
        self.initialPop = initialPop
        self.extension = extension

    # Ideas for splitting functionality

    def start(self):
        self.stats.start_timing()
        self._logger.info("Start time: %s" % (self.stats._start_time))

    def run(self):
        self.start()

        self._logger.info("*"*60)
        self._logger.info("Initializing Populations.")
        self._logger.info("*"*60)
        # Give a chance for the default confiugration
        default_conf = self.scenario.cs.get_default_configuration()
        first_pop = PopMember(default_conf, np.random.randint(self.A) + 1, 0)
        self.c_pop.append(first_pop)
        self._logger.info("First Population Member: %s" % (default_conf))

        # Initializing further 19 elements to be the initial population.
        for i in range(self.initialPop - 1):
            start_time = time.time()
            conf = self.generate_random_configuration()
            gender = np.random.randint(1000) % 2
            pop_mem = PopMember(conf, np.random.randint(self.A) + 1, gender)
            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)
            if gender == 0:
                self.insert_c(pop_mem, time_left)
            else:
                self.nc_pop.append(pop_mem)
                self._logger.info("New member added to Non-competitive Population")
                self._logger.info("New Size of NC Population: %s" % (len(self.nc_pop)))

        self._logger.info("*"*60)
        self._logger.info("Done with Initializing.")
        self._logger.info("*"*60)

        generation = 0

        while not self.stats.is_budget_exhausted():

            generation += 1
            self._logger.info("*"*60)
            self._logger.info("Generation # %s" % (generation))
            self._logger.info("*"*60)

            time_budget_left = self.scenario.wallclock_limit - (time.time() - self.stats._start_time)
            self._logger.info("Time budget left: %f" % (time_budget_left))
            # The main loop is broken when budget is exhausted.
            self.incumbent = self.c_pop[0].config
            # Best X % in C
            chosen_comp = int(math.ceil(self.X * len(self.c_pop)))
            self._logger.info("%s members of competitive Population chosen for mating" % (chosen_comp))
            best_c = self.c_pop[:chosen_comp]
            # The number of members chosen from NC
            chosen_ncomp = (200.0 / self.A) / 100.0
            chosen_ncomp *= len(self.nc_pop)
            chosen_ncomp = int(math.ceil(chosen_ncomp))
            self._logger.info("%s members of non-competitive Population chosen for mating" % (chosen_ncomp))

            # Checking if the extension applies
            if self.extension:
                self.selected_mating(best_c, chosen_ncomp)
            else:
                self.vanilla_mating(best_c, chosen_ncomp)

            # Add one more year to each population member
            for c in self.c_pop:
                c.increase_age()
            for nc in self.nc_pop:
                nc.increase_age()

            # The incumbent is first configuration in C
            self.incumbent = self.c_pop[0].config
            # Killing old members
            self.kill_old()

        self._logger.info("*"*60)
        self._logger.info("*"*60)

        self.incumbent = self.c_pop[0].config
        print("Incumbent is :")
        print(self.incumbent)
        print("With Cost")
        print(self.runhistory.get_cost(self.incumbent))
        return self.incumbent

    def vanilla_mating(self, best_c, chosen_ncomp):
        """
        The function mates each competitive member with a random
        percentage of th uncompetitive members
        """
        for c in best_c:
            # Mate each one of C with random multiple ones from NC
            start_time = time.time()
            nc = random.sample(self.nc_pop, chosen_ncomp)
            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)
            self.mate(c, nc, time_left)

    def selected_mating(self, best_c, chosen_ncomp):
        """
        The function mates each competitive member with a
        percentage of th uncompetitive members based on how
        fit the uncompetive heurstically.
        """
        for c in best_c:
            self.attractive, self.unattractive = [], []
            nc_size = len(self.nc_pop)
            # Permutation of the indexes of The NC population
            # which is changeable with each C member
            nc_perm = np.random.permutation(nc_size)
            mid = int(nc_size / 2)

            for idx in range(mid):
                first_idx = nc_perm[idx]
                second_idx = nc_perm[idx + mid]
                # Comparing pairs of NC members in parallel
                comp = threading.Thread(target=self.pairwise_comp, args=(first_idx, second_idx))
                comp.start()

            # In case of odd NC population size, the single one is
            # added to the attractive.
            if nc_size % 2 == 1:
                self.attractive.append(nc_size - 1)

            # Waiting till all threads are done
            time.sleep(3)
            # Giving the attractive members twice the chances of 
            # mating compared to the unattractive ones.
            attract_prob = 1.0 / 3.0
            nc_list = []

            # Filling the list of the NC members
            for idx in range(chosen_ncomp):
                choose_prop = np.random.random_sample()
                size_att = len(self.attractive)
                size_unatt = len(self.unattractive)

                # The condition checks if an atractive member should
                # be picked and that attractive members are not all 
                # matched.
                if choose_prop > attract_prob and size_att > 0:
                    att_idx = np.random.randint(size_att)
                    pop_num = self.attractive[att_idx]
                    del self.attractive[att_idx]
                    nc_list.append(self.nc_pop[pop_num]) 
                else:
                    unatt_idx = np.random.randint(size_unatt)
                    pop_num = self.unattractive[unatt_idx]
                    del self.unattractive[unatt_idx]
                    nc_list.append(self.nc_pop[pop_num])

                # The condition handles the rare case when an unattractive
                # member should be picked and all unattractive members are
                # all matched.
                if choose_prop <= attract_prob and size_unatt == 0:
                    att_idx = np.random.randint(size_att)
                    pop_num = self.attractive[att_idx]
                    del self.attractive[att_idx]
                    nc_list.append(self.nc_pop[pop_num])

            start_time = time.time()
            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)
            self.mate(c, nc_list, time_left)

    def pairwise_comp(self, idx1, idx2):
        """
        The method accepts 2 indexes of the NC competitives and 
        race their members configurations and send the winner to
        the attractive pile and the loser to the unattractive one.
        """
        start_time = time.time()
        first_conf = self.nc_pop[idx1].config
        second_conf = self.nc_pop[idx2].config
        time_spent = time.time() - start_time
        time_left = self._get_timebound_for_intensification(time_spent)
        winner = self.race_configs([first_conf, second_conf], time_left)
        if winner == first_conf:
            self.attractive.append(idx1)
            self.unattractive.append(idx2)
        else:
            self.unattractive.append(idx1)
            self.attractive.append(idx2)
        return

    def mutate(self, config: Configuration):
        completely_mutated = config.configuration_space.sample_configuration()
        return self.cross(completely_mutated, config, self.M)

    def insert_c(self, member, time_left):
        """
        The function uses binary search to insert a new configuration into
        readily sorted list of competitive configurations in its proper place.
        """
        lo = 0
        hi = len(self.c_pop)
        configuration = member.config
        while lo < hi:
            mid = int((lo + hi) / 2)
            list_conf = self.c_pop[mid].config

            winner = self.race_configs([list_conf, configuration], time_left)

            if winner == configuration:
                hi = mid - 1
            else:
                lo = mid + 1
        if lo == len(self.c_pop):
            self.c_pop.append(member)
        else:
            list_conf = self.c_pop[lo].config

            winner = self.race_configs([list_conf, configuration], time_left)

            if winner == configuration:
                self.c_pop.insert(lo, member)
            else:
                self.c_pop.insert(lo + 1, member)
        self._logger.info("New member added to competitive population")
        self._logger.info("New Size of C Population: %s" % (len(self.c_pop)))

        """
        for c in self.c_pop:
           print(c.config,end=',')
        print('-')
        print('-')
        print(self.runhistory.get_all_configs())
        """

    def is_young(self, member):
        # The function checks if a population member is young enough.
        return member.age <= self.A

    def kill_old(self):
        # The function kills all old members except the incumbent
        first_c = self.c_pop[0]
        c_pop_cpy = self.c_pop[1:]
        c_pop_cpy = [pop for pop in c_pop_cpy if self.is_young(pop)]
        c_pop_cpy.insert(0, first_c)
        self.c_pop = c_pop_cpy
        self.nc_pop = [pop for pop in self.nc_pop if self.is_young(pop)]
        self._logger.info("Killed old members")
        self._logger.info("New Size of NC Population: %s" % (len(self.nc_pop)))
        self._logger.info("New Size of C Population: %s" % (len(self.c_pop)))

    def generate_random_configuration(self):
        """
        The function generates random configurations by randomizing each
        hyparameters in the configuration until a valid configuration is
        returned.
        """
        hyper_params = self.scenario.cs.get_hyperparameters()
        candidate_conf = None
        found = False
        while not found:
            values = dict()
            for param in hyper_params:
                key, val = self.random_parameter(param)
                values[key] = val
            try:
                candidate_conf = Configuration(configuration_space=self.scenario.cs,
                                               values=values,
                                               allow_inactive_with_values=True)
                found = True
            except ValueError as e:
                continue
        return candidate_conf

    def random_parameter(self, param):
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
            idx = np.random.random_integers(0, param._num_choices)
            choices_list = list(choices)
            val = choices_list[idx]
            return param.name, val
        else:
            idx = np.random.random_integers(0, param._num_choices)
            choices_list = list(sequence)
            val = choices_list[idx]
            return param.name, val

    def cross(self, parent1, parent2, percent):
        """Crosses two configuration and creates a new one.
        With probability percent choose value from parent1 for each
        hyperparameter that is not dependent on any other,
        then also include children of this hp from same parent.
        """
        cs = self.scenario.cs
        hyp_names = cs.get_hyperparameter_names()

        new_values = dict()

        for name in hyp_names:
            # print("Hyperparameter: %s" % (name))
            if name not in new_values.keys() and cs.get_parents_of(name) == []:
                ran = np.random.randint(100)
                if ran % math.ceil(100 / percent) == 0:
                    parent = parent1.get_dictionary()
                else:
                    parent = parent2.get_dictionary()

                # print("Chosen parent: %s" % (ran % 2))
                val = parent[name]
                new_values[name] = val

                childs = cs.get_children_of(name)
                # print("Children of %s: %s" % (name, childs))
                for child in childs:
                    # this condition can be excluded
                    if child.name in parent.keys():
                        new_values[child.name] = parent[child.name]

        config = Configuration(cs, values=new_values)
        return config

    def mate(self, c, nc_list, time_left):
        """ Mates one competitive member with set
        of non-competitive member and for produces a child , 
        mutate it and add it with age 0 and random gender to its population set.
        """

        for nc in nc_list:
            
            # cross both partner with probability of 50%, get new config
            new_conf = self.cross(c.config, nc.config, 50)

            # mutate configuration
            new_conf = self.mutate(new_conf)

            # create new child, randomly set gender
            g = np.random.randint(0, 1000)
            if g % 2 == 0:
                child = PopMember(new_conf, 0, 0)
                self.insert_c(child, time_left)
            else:
                child = PopMember(new_conf, 0, 1)
                self.nc_pop.append(child)
        self._logger.info("Mated one member of competitive population with %s members of non-competitive population." % (len(nc_list)))

    def race_configs(self, set_of_conf, time_left):
        """Races the challengers agains each other to determine incumbent

        """
        # print("Time left: %s" % (max(self.intensifier._min_time, time_left)))
        try:
            best, inc_perf = self.intensifier.intensify(
                challengers=[set_of_conf[1]],
                incumbent=set_of_conf[0],
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
                time_bound=max(self.intensifier._min_time, time_left)
                # time_bound=10
            )
            return best
        except:
            return
        # print("Incumbent: %s" % (self.incumbent))
        # print("Performance: %s" % (inc_perf))

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

        # print("Total time: %.4f, time spent on choosing next "
        #                  "configurations: %.4f (%.2f), time left for "
        #                  "intensification: %.4f (%.2f)" %
        #                  (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))

        # self.logger.debug("Total time: %.4f, time spent on choosing next "
        #                   "configurations: %.4f (%.2f), time left for "
        #                   "intensification: %.4f (%.2f)" %
        #                   (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left

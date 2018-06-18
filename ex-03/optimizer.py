import numpy as np
import time
import math
import random

from smac.configspace import Configuration, ConfigurationSpace
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from ConfigSpace.util import get_random_neighbor
from ConfigSpace.hyperparameters import Hyperparameter
from PopMember import PopMember


class ESOptimizer(object):

    # ES Parameters
    X = .1
    P = .5
    M = .1
    A = 3
    S = .1

    """Interface that contains the main Evolutionary optimizer

    Attributes:

    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 rng: np.random.RandomState):

        self.incumbent = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.nc_pop = []
        self.c_pop = []

    # Ideas for splitting functionality

    def start(self):
        self.stats.start_timing()

    def run(self):
        self.start()
        # Give a chance for the default confiugration
        default_conf = self.config_space.get_default_confiugration()
        first_pop = PopMember(config: default_conf, 
                              age: np.random.randint(A) + 1,
                              0)
        self.c_pop.append(default_conf)

        # Initializing further 19 elements to be the initial population.
        for i in range(19):
            start_time = time.time()
            conf = self.generate_random_configuration()
            gender = np.random.randint(1000) % 2
            pop_mem = PopMember(config: conf, 
                                age: np.random.randint(A) + 1,
                                gender)
            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)
            if gender == 0:
                self.insert_c(pop_mem, time_left)
            else:
                self.nc_pop.append(pop_mem)

        while True:
            # The main loop is broken when budget is exhausted.
            self.incumbent = self.c_pop[0].config
            # Best X % in C
            chosen_comp = int(math.ceil(X * len(self.c_pop)))
            best_c = self.c_pop[:chosen_comp]
            # The number of members chosen from NC
            chosen_ncomp = (200.0 / A) / 100.0
            chosen_ncomp *= len(self.nc_pop)
            chosen_ncomp = int(math.ceil(chosen_ncomp))
            for c in best_c:
                # Mate each one of C with random multiple ones from NC
                start_time = time.time()
                nc = random.sample(self.nc_pop, chosen_ncomp)
                time_spent = time.time() - start_time
                time_left = self._get_timebound_for_intensification(time_spent)
                self.mate(c, nc, time_left)
            # Add one more year to each population member
            """
            for c in self.c_pop:
                c.add_year()
            for nc in self.nc_pop:
                nc.add_year()
            """
            # The incumbent is first configuration in C
            self.incumbent = self.c_pop[0].config
            # Killing old members
            self.kill_old()

            if self.stats.is_budget_exhausted():
                break

        return self.incumbent




    def mutate(self, config):
        pass

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
            if mid == 0:
                winner = self.race_configs([list_conf, configuration], time_left)
                if winner == configuration:
                    self.c_pop.insert(0, member)
                    return
                else:
                    self.c_pop.insert(1, member)
                    return
            elif mid == len(self.c_pop) - 1:
                winner = self.race_configs([list_conf, configuration], time_left)
                if winner == configuration:
                    self.c_pop.insert(len(self.c_pop) - 1, member)
                    return
                else:
                    self.c_pop.append(member)
                    return
            else:
                winner = self.race_configs([list_conf, configuration], time_left)
                if winner == configuration:
                    lo = mid
                else:
                    hi = mid
        list_conf = self.c_pop[lo].config
        winner = self.race_configs([list_conf, configuration], time_left)
        if winner == configuration:
            self.c_pop.insert(lo, member)
        else:
            self.c_pop.insert(lo + 1, member)


    def is_young(self, member):
        # The function checks if a population member is young enough.
        return member.age <= A

    def kill_old(self):
        # The function kills all old members except the incumbent
        first_c = self.c_pop[0]
        c_pop_cpy = self.c_pop[1:]
        c_pop_cpy = [pop for pop in c_pop_cpy if self.is_young(pop)]
        c_pop_cpy.insert(0, first_c)
        self.c_pop = c_pop_cpy
        self.nc_pop = [pop for pop in self.nc_pop if self.is_young(pop)]



        
    def generate_random_configuration(self):
        """ 
        The function generates random configurations by randomizing each 
        hyparameters in the configuration until a valid configuration is 
        returned.
        """
        hyper_params = self.config_space.get_hyperparameters()
        candidate_conf = None
        found = False
        while not found:
            values = dict()
            for param in hyper_params:
                key, val = self.random_parameter(param)
                values[key] = val
            try:
                candidate_conf = Configuration(configuration_space: self.config_space
                    ,values: values
                    ,allow_inactive_with_values: True)
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

    def get_parents(self):
        pass

    def cross(self, parent1, parent2):
        """Crosses two configuration and creates a new one.
        Randomly select value from either parent for each
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
                if ran % 2 == 0:
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
            
            # cross both partner, get new config
            new_conf = self.cross(c.config, nc.config)

            # mutate configuration
            # new_conf = mutate(new_conf)

            # create new child, randomly set gender
            g = np.random.randint(0, 1000)
            if g % 2 == 0:
                child = PopMember(new_conf, 0, 0)
                self.insert_c(child, time_left)
            else:
                child = PopMember(new_conf, 0, 1)
                self.nc_pop.append(child)

    def race_configs(self, set_of_conf, time_left):
        """Races the challengers agains each other to determine incumbent

        """
        # print("Time left: %s" % (max(self.intensifier._min_time, time_left)))

        best, inc_perf = self.intensifier.intensify(
            challengers=[set_of_conf[1]],
            incumbent=set_of_conf[0],
            run_history=self.runhistory,
            aggregate_func=self.aggregate_func,
            time_bound=max(self.intensifier._min_time, time_left)
            # time_bound=10
        )
        return best
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

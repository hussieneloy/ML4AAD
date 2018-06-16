import numpy as np
import time

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
        start_time = time.time()

        # using two random configs here for testing
        conf1 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(42))
        conf2 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(24))
        conf3 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(32))
        conf4 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(23))

        nc_list = [PopMember(conf1, 0, 1), PopMember(conf2, 0, 1)]
        c_list = [PopMember(conf3, 0, 0), PopMember(conf4, 0, 0)]

        time_spent = time.time() - start_time
        time_left = self._get_timebound_for_intensification(time_spent)

        # find two random configs, needs to be replaced with function get next configs
        print("Conf1: %s" % (conf1))
        print("Conf2: %s" % (conf2))

        # self.race_configs([conf1, conf2], time_left)
        # incumbent = self.cross(conf1, conf2)
        # print(incumbent)
        self.mate(nc_list, c_list)

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

    def mate(self, nc_list, c_list):
        """ Mates two sets of population members
        Mate every member from non-competitive set with one from competitive set.
        Then cross these parents to generate a new configuration, mutate it and
        add it with age 0 and random gender to its population set.
        """
        childs = []

        for nc in nc_list:
            # get partner randomly
            c = c_list[np.random.randint(0, len(c_list))]

            # cross both partner, get new config
            new_conf = self.cross(c.config, nc.config)

            # mutate configuration
            # new_conf = mutate(new_conf)

            # create new child, randomly set gender
            g = np.random.randint(0, 1000)
            if g % 2 == 0:
                child = PopMember(new_conf, 0, 0)
                self.c_pop.append(child)
            else:
                child = PopMember(new_conf, 0, 1)
                self.nc_pop.append(child)

    def race_configs(self, set_of_conf, time_left):
        """Races the challengers agains each other to determine incumbent

        """
        # print("Time left: %s" % (max(self.intensifier._min_time, time_left)))

        self.incumbent, inc_perf = self.intensifier.intensify(
            challengers=set_of_conf,
            incumbent=self.incumbent,
            run_history=self.runhistory,
            aggregate_func=self.aggregate_func,
            time_bound=max(self.intensifier._min_time, time_left)
            # time_bound=10
        )
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

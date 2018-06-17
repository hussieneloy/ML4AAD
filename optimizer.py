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


class ES(object):

    """Interface that contains the main Evolutionary optimizer

    Attributes:

    """

    def __init__(self,
                 scenario: Scenario,
                 runhistory: RunHistory,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration=None):
        
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

    def run(self):

        conf1 = get_random_neighbour(self.scenario.cs.get_default_configuration())
        conf2 = get_random_neighbour(self.scenario.cs.get_default_configuration())

        print("Conf1: %s" % (conf1))
        print("Conf2: %s" % (conf2))

        self.race_configs([conf1, conf2])

        print(self.incumbent)

    def get_parents(self):
        pass

    def cross_over(self, parent1, parent2):
        pass

    def age_test(self, threshold, nc_list, c_list):
        # maybe using c_pop and nc_pop as parameter
        for mem in nc_list:
            mem.increase_age()
            if(mem.age >= threshold):
                if(mem != self.incumbent):
                    self.nc_pop.remove(mem)
        for cmem in c_list:
            cmem.increase_age()
            if(cmem.age >= threshold):
                if(cmem != self.incumbent):
                    self.nc_pop.remove(cmem)

    def race_configs(self, set_of_conf):
        """Races the challengers agains each other to determine incumbent

        """
        time_left = 5

        self.incumbent, inc_perf = self.intensifier.intensify(
            challengers=set_of_conf,
            incumbent=self.incumbent,
            run_history=self.runhistory,
            aggregate_func=self.aggregate_func,
            time_bound=max(self.intensifier._min_time, time_left)
        )

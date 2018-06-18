import numpy as np

from smac.configspace import Configuration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
import ConfigSpace.util
from smac.tae.execute_ta_run import FirstRunCrashedException


class ESOptimizer(object):

    """Interface that contains the main Evolutionary optimizer

    Attributes:

    """

    def __init__(self,
                 scenario: Scenario,
                 runhistory: RunHistory,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration=None,
                 X: int=10,
                 M: int=10,
                 A: int=3):
        
        self.incumbent = restore_incumbent
        self.scenario = scenario
        self.config_space = scenario.cs
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.X = X
        self.M = M
        self.A = A

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

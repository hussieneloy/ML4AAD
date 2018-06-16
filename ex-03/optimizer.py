import numpy as np
import time

from smac.configspace import Configuration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from ConfigSpace.util import get_random_neighbor


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

    # Ideas for splitting functionality

    def start(self):
        self.stats.start_timing()

    def run(self):
        start_time = time.time()

        # using two random configs here for testing
        conf1 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(42))
        conf2 = get_random_neighbor(self.scenario.cs.get_default_configuration(), np.random.randint(24))

        time_spent = time.time() - start_time
        time_left = self._get_timebound_for_intensification(time_spent)

        # find two random configs, needs to be replaced with function get next configs
        print("Conf1: %s" % (conf1))
        print("Conf2: %s" % (conf2))

        self.race_configs([conf1, conf2], time_left)

        print(self.incumbent)

    def get_parents(self):
        pass

    def cross_over(self, parent1, parent2):
        pass

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

import os
import sys
import logging
import numpy as np
import shutil

from utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from es_facade import ES
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
# from smac.optimizer.objective import average_cost
# from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
# from smac.utils.io.input_reader import InputReader
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException
# from smac.utils.io.output_directory import create_output_directory


class ESCLI(object):

    """Main class of ES"""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def main_cli(self):
        """ Main function of ES for CLI interface"""
        self.logger.info("ES call: %s" % (" ".join(sys.argv)))

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)
        logger_handler = logging.StreamHandler(
                stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter(
                "%(levelname)s:\t%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                "%Y-%m-%d %H:%M:%S")
        logger_handler.setFormatter(formatter)
        root_logger.addHandler(logger_handler)
        # remove default handler
        root_logger.removeHandler(root_logger.handlers[0])

        # Create scenario-object
        scen = Scenario(args_.scenario_file, misc_args)

        # Create optimizer
        optimizer = ES(
            scenario=scen,
            rng=np.random.RandomState(args_.seed),
            run_id=args_.seed,
            parallel_options=args_.parallel_options,
            cores=args_.cores
        )

        try:
            self.logger.info("Started Optimization")
            optimizer.optimize()
        except (TAEAbortException, FirstRunCrashedException) as err:
            self.logger.error(err)

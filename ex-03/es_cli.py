import os
import sys
import logging
import numpy as np
import shutil

from utils.io.cmd_reader import CMDReader
# from smac.scenario.scenario import Scenario
from es_facade import ES
# from smac.runhistory.runhistory import RunHistory
# from smac.stats.stats import Stats
# from smac.optimizer.objective import average_cost
# from smac.utils.merge_foreign_data import merge_foreign_data_from_file
# from smac.utils.io.traj_logging import TrajLogger
# from smac.utils.io.input_reader import InputReader
# from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException
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
        pass

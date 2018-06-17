from collections import defaultdict
import os
import sys
import logging
import numpy
import shlex
import time
import datetime
import copy
import typing

# from smac.utils.io.input_reader import InputReader
# from smac.utils.io.output_writer import OutputWriter
# from smac.utils.constants import MAXINT
# from smac.configspace import pcs, pcs_new

""" 
def _is_truthy(arg):
    if arg in ["1", "true", "True", True]:
        return True
    elif arg in ["0", "false", "False", False]:
        return False
    else:
        raise ValueError("{} cannot be interpreted as a boolean argument. "
"Please use one of {{0, false, 1, true}}.".format(arg))
"""

class Scenario(object):
    """
    Scenario contains the configuration of the optimization process and
    constructs a scenario object from a file or dictionary.

    All arguments set in the Scenario are set as attributes.
    """

    def __init__(self, scenario, cmd_args: dict=None):
        """ Creates a scenario-object. The output_dir will be
        "output_dir/run_id/" and if that exists already, the old folder and its
        content will be moved (without any checks on whether it's still used by
        another process) to "output_dir/run_id.OLD". If that exists, ".OLD"s
        will be appended until possible.

        Parameters
        ----------
        scenario : str or dict
            If str, it will be interpreted as to a path a scenario file
            If dict, it will be directly to get all scenario related information
        cmd_args : dict
            Command line arguments that were not processed by argparse
        """
        pass









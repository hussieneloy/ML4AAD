
import os
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS


# X [1,100] [10]
# M [1,100] [10]
# A [1, 10] [3]


class CMDReader(object):

    """Use argparse to parse command line options
    Attributes
    ----------
    logger : Logger
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        pass

    def read_cmd(self):
        """Reads command line options
        Returns
        -------
            args_: parsed arguments; return of parse_args of ArgumentParser
        """

        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=1, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        req_opts.add_argument("--X",
                              help="percentage, select top X percent of competitive for mating",
                              type=int,
                              default=10)
        req_opts.add_argument("--M",
                              help="percentage, with prop M variable of config is mutated",
                              type=int,
                              default=10)
        req_opts.add_argument("--A",
                              help="max. age of population member",
                              type=int,
                              default=3)
        req_opts.add_argument("--initialPop",
                              help="number of initial Population, half will be competitive and half non-competitive",
                              type=int,
                              default=20)
        req_opts.add_argument("--extension",
                              help="using extension with sexual selection if true, else standard selection",
                              type=bool,
                              default=False)

        # deleted everything with warmstart (not needed)
        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    def _check_args(self, args_):
        """Checks command line arguments (e.g., whether all given files exist)
        Parameters
        ----------
        args_: parsed arguments
            Parsed command line arguments
        Raises
        ------
        ValueError
            in case of missing files or wrong configurations
        """
        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))

        print(args_)

        if args_.X < 1 or args_.X > 100:
            raise ValueError("X is an percentage and must be in [1, 100]")
        if args_.M < 1 or args_.M > 100:
            raise ValueError("M is an percentage and must be in [1, 100]")
        if args_.A < 1:
            raise ValueError("Killing age must be a positive integer")

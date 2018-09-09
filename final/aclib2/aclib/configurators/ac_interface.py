from aclib.configurators.smac2 import SMAC2
from aclib.configurators.smac3 import SMAC3
from aclib.configurators.paramils import ParamILS
from aclib.configurators.gga import GGA
from aclib.configurators.ggapp import GGAPP
from aclib.configurators.roar import ROAR
from aclib.configurators.irace2 import IRACE2
from aclib.configurators.tool_cl_each import TOOL_CL_EACH
from aclib.configurators.tool_cl_list import TOOL_CL_LIST
from aclib.configurators.tool_ef_each import TOOL_EF_EACH
from aclib.configurators.tool_ef_list import TOOL_EF_LIST
from aclib.configurators.tool_ucb_each import TOOL_UCB_EACH
from aclib.configurators.tool_ucb_list import TOOL_UCB_LIST

__author__ = "Marius Lindauer"
__version__ = "0.0.1"
__license__ = "BSD"


class ACInterface(object):

    def __init__(self, aclib_root: str, suffix_dir: str=""):
        '''
            Constructor

            Arguments
            ---------
            aclib_root: str
                root directory to AClib
            suffix_dir : str
                suffix of AC procedure directory
        '''

        self.aclib_root = aclib_root
        self.suffix_dir = suffix_dir

    def get_AC(self, ac_name: str):
        '''
            returns AC interface object

            Arguments
            ---------
            ac_name: str
                name of algorithm configurator
        '''

        if ac_name == "SMAC2":
            ac = SMAC2(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "SMAC3":
            ac = SMAC3(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "PARAMILS":
            ac = ParamILS(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "GGA":
            ac = GGA(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "GGA++":
            ac = GGAPP(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "GGA_def":
            ac = GGA(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir, ie_def=True)
        elif ac_name == "GGA++_def":
            ac = GGAPP(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir, ie_def=True)
        elif ac_name == "ROAR":
            ac = ROAR(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "IRACE2":
            ac = IRACE2(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_CL_EACH":
            ac = TOOL_CL_EACH(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_CL_LIST":
            ac = TOOL_CL_LIST(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_EF_EACH":
            ac = TOOL_EF_EACH(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_EF_LIST":
            ac = TOOL_EF_LIST(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_UCB_EACH":
            ac = TOOL_UCB_EACH(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        elif ac_name == "TOOL_UCB_LIST":
            ac = TOOL_UCB_LIST(aclib_root=self.aclib_root, suffix_dir=self.suffix_dir)
        else:
            raise ValueError("Unknow Configurator: %s" %(ac_name))

        return ac

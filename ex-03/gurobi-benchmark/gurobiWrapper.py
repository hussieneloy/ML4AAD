#!/usr/bin/env python2.7
# encoding: utf-8

'''
emptyWrapper -- template for a wrapper based in genericWrapper.py

@author:     Marius Lindauer, Chris Fawcett, Alex Fréchette, Frank Hutter
@copyright:  2014 AClib. All rights reserved.
@license:    GPL
@contact:    lindauer@informatik.uni-freiburg.de, fawcettc@cs.ubc.ca, afrechet@cs.ubc.ca, fh@informatik.uni-freiburg.de

'''

from genericWrapper import AbstractWrapper
import os
import inspect
import sys

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, "../.."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)



class GurobiWrapper(AbstractWrapper):
    
    def get_command_line_args(self, runargs, config):
        '''
        Returns the command line call string to execute the target algorithm (here: Spear).
        Args:
            runargs: a map of several optional arguments for the execution of the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,
                      "runlength" : <runlength cutoff>,
                      "seed" : <seed>
                    }
            config: a mapping from parameter name to parameter value
        Returns:
            A command call list to execute the target algorithm.
        '''
        binary_path = cmd_folder + "/solver-binary/gurobi/gurobi_cl"
        cmd = "%s" %(binary_path)

        for name, value in config.items():
            name_mod = name.replace("-", "")
            cmd += " %s=%s" %(name_mod, value)
        cmd += " TimeLimit=%s" %(runargs["cutoff"]-1)
        cmd += " Threads=1"
	
        cmd += " %s" %(runargs["instance"])
        return cmd
    
    def process_results(self, filepointer, out_args):
        '''
        Parse a results file to extract the run's status (SUCCESS/CRASHED/etc) and other optional results.
    
        Args:
            filepointer: a pointer to the file containing the solver execution standard out.
            out_args : a map with {"exit_code" : exit code of target algorithm} 
        Returns:
            A map containing the standard AClib run results. The current standard result map as of AClib 2.06 is:
            {
                "status" : <"SUCCESS"/"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "runtime" : <runtime of target algrithm>,
                "quality" : <a domain specific measure of the quality of the solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with the run [optional]>
            }
            ATTENTION: The return values will overwrite the measured results of the runsolver (if runsolver was used). 
        '''
        import re

        data = filepointer.read()
        resultMap = {}

        # status
        if (re.search('Solved in [.]+?\nOptimal objective', data)):
            resultMap["status"]= "SUCCESS"
        elif (re.search('Optimal solution found', data)):
            resultMap["status"]= "SUCCESS"
        elif (re.search('Stopped in [. ]+?\nSolve interrupted', data)):
            resultMap["status"] = "ABORT"
        elif (re.search("Error", data)):
            resultMap["status"] = "CRASHED"
        elif (re.search('Time limit reached\n', data)):
            resultMap["status"] = "TIMEOUT"
        else:
            pass



        # runtime
        if (re.search('Solved in [0-9]+? iterations and ([0-9]+?.[0-9]+?) seconds', data)):
            runtime = float(re.search('Solved in [0-9]+? iterations and ([0-9]+?.[0-9]+?) seconds', data).group(1))
            resultMap["runtime"] = runtime

        elif (re.search('Stopped in [0-9]+? iterations and ([0-9]+?.[0-9]+?) seconds', data)):
            runtime = float(re.search('Stopped in [0-9]+? iterations and ([0-9]+?.[0-9]+?) seconds', data).group(1))
            resultMap["runtime"] = runtime

        elif (re.search('Explored [0-9]+? nodes [(][0-9]+? simplex iterations[)] in ([0-9]+?.[0-9]+?) seconds', data)):
            runtime = float(re.search('Explored [0-9]+? nodes [(][0-9]+? simplex iterations[)] in ([0-9]+?.[0-9]+?) seconds', data).group(1))
            resultMap["runtime"] = runtime

        # quality
        if (re.search('Best objective .+?, best bound .+?, gap [0-9]+?.[0-9]+?%', data)):
            gap = float(re.search('Best objective .+?, best bound .+?, gap ([0-9]+?.[0-9]+?)%', data).group(1))
            resultMap["quality"] = gap


        print(resultMap)
        return resultMap

if __name__ == "__main__":
    wrapper = GurobiWrapper()
    wrapper.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from environment import Environment
from onlineSolver import OnlineSolver
from offlineSolver import OfflineSolver
from policyReader import PolicyReader
from aems import AEMS2
from mdpSolver import QMDP, MinMDP

import sys

if len(sys.argv) == 5:
    #offline solver
    model_name = sys.argv[2]
    model_file = 'examples/env/' + model_name + '.pomdp'
    pomdp  = POMDP(model_file)
    num_runs = int(sys.argv[3])
    precision = float(sys.argv[4])
    if sys.argv[1] == "QMDP":
        solver = QMDP(pomdp, precision)
    elif sys.argv[1] == "MinMDP":
            solver = MinMDP(pomdp, precision)
    else:
        raise Exception("Invalid offline solver: ", sys.argv[1])
    print ("Average reward: ", solver.evaluate(num_runs))
    
elif len(sys.argv) == 8:
    #online solver
    if sys.argv[1] != "AEMS2":
        raise Exception("Invalid online solver: ", sys.argv[1])
    
    model_name = sys.argv[5]
    model_file = 'examples/env/' + model_name + '.pomdp'
    pomdp  = POMDP(model_file)
    precision = float(sys.argv[7])
    if sys.argv[2] == "MinMDP":
        lb = MinMDP(pomdp, precision)
    elif sys.argv[2] == "PBVI":
        policy_file = 'examples/policy/' + model_name + '.policy'   # currently only for 3 problems
        lb = PolicyReader(pomdp, policy_file)
    else:
        raise Exception("Invalid lower bound:", sys.argv[2])
    
    if sys.argv[3] != "QMDP":
        raise Exception("Invalid higher bound solver:", sys.argv[3])
    
    ub = QMDP(pomdp, precision)
    pomdp  = POMDP(model_file)
    time_limit = float(sys.argv[4])
    num_runs = int(sys.argv[6])
    total_reward = 0
    for runs in range(num_runs):    
        solver = AEMS2(pomdp, lb, ub, precision, time_limit)
        total_reward += OnlineSolver.solve(solver)
    print ("Average reward: ", total_reward/num_runs)
else:
    raise ("Invalid format")
    
    
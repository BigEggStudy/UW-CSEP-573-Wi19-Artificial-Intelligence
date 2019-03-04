#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from policyReader import PolicyReader
from onlineSolver import OnlineSolver
from aems import AEMS2
from mdpSolver import QMDP, MinMDP

import sys
import numpy as np

class Test:
    @staticmethod 
    def QMDPTest():
        score = 0
        f = open('tests/QMDP', 'r')
        contents = [x.strip() for x in f.readlines() if (not (x.isspace()))]
        i = 0
        while i < len(contents):
            line = contents[i]
            if line.startswith('#'):
                print (line)
            elif line.startswith('Environment'):
                model_name = line.split()[1]
                model_file = 'examples/env/' + model_name + '.pomdp'
                print ('Environment:', model_name)
                pomdp  = POMDP(model_file)
                qmdp = QMDP(pomdp, .01)
            elif line.startswith('Belief'):
                pieces = [x for x in line.split() if (x.find(':') == -1)]
                belief = np.array([float(x) for x in pieces])
                print ('Belief =', belief)
            elif line.startswith('Value'):
                value = float(line.split()[1])
                ans_value = qmdp.getValue(belief)
                print ("Value by QMDP:" , value , "Your answer:", ans_value)
            elif line.startswith('Action'):
                action = int(line.split()[1])
                ans_action = qmdp.chooseAction(belief)
                print ("Action by QMDP:" , action , "Your answer:", ans_action)
                if abs(ans_value - value) < .01 and action == ans_action:
                    score += 1
                    print ("PASS")
                else:
                    print ("FAIL")
            elif line.startswith('Runs'):
                num_runs = int(line.split()[1])
                ans_total_reward = qmdp.evaluate(num_runs)
            elif line.startswith('Reward'):
                total_reward = float(line.split()[1])
                print ("Reward by QMDP:" , total_reward , "Your answer:", ans_total_reward)
            elif line.startswith('Error'):
                error = float(line.split()[1])
                if abs(total_reward - ans_total_reward) < error:
                    score +=1
                    print ("PASS")
                else:
                    print("FAIL")
            else:
                raise Exception("Unrecognized line: " + line)
        
            i +=1
        print ("Total score out of 5:", score)
        return score
    
    @staticmethod 
    def MinMDPTest():
        score = 0
        f = open('tests/MinMDP', 'r')
        contents = [x.strip() for x in f.readlines() if (not (x.isspace()))]
        i = 0
        while i < len(contents):
            line = contents[i]
            if line.startswith('#'):
                print (line)
            elif line.startswith('Environment'):
                model_name = line.split()[1]
                model_file = 'examples/env/' + model_name + '.pomdp'
                print ('Environment:', model_name)
                pomdp  = POMDP(model_file)
                min_mdp = MinMDP(pomdp, .01)
            elif line.startswith('Belief'):
                pieces = [x for x in line.split() if (x.find(':') == -1)]
                belief = np.array([float(x) for x in pieces])
                print ('Belief =', belief)
            elif line.startswith('Value'):
                value = float(line.split()[1])
                ans_value = min_mdp.getValue(belief)
                print ("Value by MinMDP:" , value , "Your answer:", ans_value)
            elif line.startswith('Action'):
                action = int(line.split()[1])
                ans_action = min_mdp.chooseAction(belief)
                print ("Action by MinMDP:" , action , "Your answer:", ans_action)
                if abs(ans_value - value) < .01 and action == ans_action:
                    score += 1
                    print ("PASS")
                else:
                    print ("FAIL")
            elif line.startswith('Runs'):
                num_runs = int(line.split()[1])
                ans_total_reward = min_mdp.evaluate(num_runs)
            elif line.startswith('Reward'):
                total_reward = float(line.split()[1])
                print ("Reward by MinMDP:" , total_reward , "Your answer:", ans_total_reward)
            elif line.startswith('Error'):
                error = float(line.split()[1])
                if abs(total_reward - ans_total_reward) < error:
                    score +=1
                    print ("PASS")
                else:
                    print ("FAIL")
            else:
                raise Exception("Unrecognized line: " + line)
        
            i +=1
        print ("Total score out of 3:", score)
        return score
    
    @staticmethod 
    def AEMS2Test():
        score = 0
        f = open('tests/AEMS2', 'r')
        contents = [x.strip() for x in f.readlines() if (not (x.isspace()))]
        i = 0
        while i < len(contents):
            line = contents[i]
            if line.startswith('#'):
                print (line)
            elif line.startswith('Environment'):
                model_name = line.split()[1]
                print ('Environment:', model_name)
                model_file = 'examples/env/' + model_name + '.pomdp'
                pomdp  = POMDP(model_file)
                qmdp = QMDP(pomdp, .01)
                m_mdp = MinMDP(pomdp, .01)
            elif line.startswith('Time'):
                time_limit = float(line.split()[1])
                print ("Time limit (sec):" , time_limit)
            elif line.startswith('Runs'):
                num_runs = int(line.split()[1])
                sum_reward = 0
                for run in range(num_runs):
                    solver = AEMS2(pomdp, m_mdp, qmdp, .01, time_limit)
                    sum_reward += OnlineSolver.solve(solver)
                ans_total_reward = sum_reward / num_runs
                print (ans_total_reward)
            elif line.startswith('Reward'):
                total_reward = float(line.split()[1])
                print ("Minimum requried reward:" , total_reward , "Your answer:", ans_total_reward)
                if total_reward <= ans_total_reward:
                    score +=4
                    print ("PASS")
                else:
                    print ("FAIL")
            else:
                raise Exception("Unrecognized line: " + line)
            i +=1
        print ("Total score out of 8:", score)
        return score
    
    @staticmethod
    def allTests():
        score = 0 
        score += Test.QMDPTest()
        score += Test.MinMDPTest()
        score += Test.AEMS2Test()
        print ("Overall score out of 16:", score)
        
        
if sys.argv[1] == 'q2':
    Test.QMDPTest()

elif sys.argv[1] == 'q3':
    Test.MinMDPTest()
    
elif sys.argv[1] == 'q4':
    Test.AEMS2Test()
elif sys.argv[1] == 'all':
    Test.allTests()
else:
    raise Exception("Invalid Request")

